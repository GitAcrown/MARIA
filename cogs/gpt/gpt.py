import asyncio
import io
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Literal

import discord
import pytz
import tiktoken
import unidecode
from discord import Interaction, app_commands
from discord.ext import commands
from moviepy.editor import VideoFileClip
from openai import AsyncOpenAI

from common import dataio
from common.utils import fuzzy, pretty

logger = logging.getLogger(f'MARIA.{__name__.split(".")[-1]}')

FULL_SYSTEM_PROMPT = lambda data: f"""# FONCTIONNEMENT INTERNE
La discussion se déroule sur un salon textuel Discord dont tu disposes de l'historique des messages.
Les messages sont précédés du nom de l'utilisateur qui les a envoyés. Tu ne met pas ton nom devant tes réponses.
Tu es capable de voir les images que les utilisateurs envoient.
Tu disposes de trois fonctions te permettant de gérer des notes sur les utilisateurs, consulte-les quand tu as besoin d'infos sur un utilisateur.
Tu suis scrupuleusement les instructions ci-après.

# INFORMATIONS
SALON : {data['channel_name']}
SERVEUR : {data['guild_name']}
TON NOM : {data['bot_name']}
DATE/HEURE : {data['current_date']}

# INSTRUCTIONS
{data['system_prompt']}"""
DEFAULT_SYSTEM_PROMPT = "Tu es un assistant utile et familier qui répond aux questions des différents utilisateurs de manière concise et simple."
MAX_COMPLETION_TOKENS = 300
MAX_CONTEXT_TOKENS = 4096
ENABLE_TOOL_USE : bool = True

GPT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_user_info",
            "description": "Renvoie une information associée à un utilisateur.",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["user", "key"],
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "Le nom de l'utilisateur dont on veut récupérer les notes.",
                    },
                    "key": {
                        "type": "string",
                        "description": "La clé de la note à récupérer (ex. 'age', 'ville', etc.). Renvoie l'information de la clé la plus proche si la clé exacte n'est pas trouvée.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_all_user_info",
            "description": "Renvoie toutes les informations associées à un utilisateur.",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["user"],
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "Le nom de l'utilisateur dont on veut récupérer les notes.",
                    }
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_user_info",
            "description": "Modifie une information associée à un utilisateur.",
            "strict": True,
            "parameters": {
                "type": "object",
                "required": ["user", "key", "value"],
                "properties": {
                    "user": {
                        "type": "string",
                        "description": "Le nom de l'utilisateur dont on veut modifier les notes.",
                    },
                    "key": {
                        "type": "string",
                        "description": "La clé de la note à modifier (ex. 'age', 'ville', etc.).",
                    },
                    "value": {
                        "type": "string",
                        "description": "La nouvelle valeur de la note.",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]

def clean_name(name: str) -> str:
    name = ''.join([c for c in unidecode.unidecode(name) if c.isalnum() or c.isspace()]).rstrip()
    return re.sub(r"[^a-zA-Z0-9_-]", "", name[:32])

class SystemPromptModal(discord.ui.Modal, title="Modifier les instructions système"):
    """Modal pour modifier ou consulter le prompt du système."""
    def __init__(self, current_system_prompt: str):
        super().__init__(timeout=None)
        self.current_system_prompt = current_system_prompt
        
        self.new_system_prompt = discord.ui.TextInput(
            label="Instructions système",
            style=discord.TextStyle.long,
            placeholder="Instructions de fonctionnement de l'assistant",
            default=self.current_system_prompt,
            required=True,
            min_length=10,
            max_length=500
        )
        self.add_item(self.new_system_prompt)
        
    async def on_submit(self, interaction: Interaction) -> None:
        await interaction.response.defer(ephemeral=True)
        return self.stop()
        
    async def on_error(self, interaction: Interaction, error: Exception) -> None:
        return await interaction.response.send_message(f"**Erreur** × {error}", ephemeral=True)
    
class ConfirmView(discord.ui.View):
    """Permet de confirmer une action."""
    def __init__(self, author: discord.Member | discord.User):
        super().__init__()
        self.value = False
        self.author = author
    
    @discord.ui.button(label="Confirmer", style=discord.ButtonStyle.success)
    async def confirm(self, interaction: Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        self.value = True
        self.stop()
    
    @discord.ui.button(label="Annuler", style=discord.ButtonStyle.danger)
    async def cancel(self, interaction: Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        self.value = False
        self.stop()

class MessageContentElement:
    def __init__(self, type: Literal['text', 'image_url'], raw_content: str):
        self.type = type
        self.raw_content = raw_content
    
    def __repr__(self):
        return f'<MessageContentElement type={self.type} raw_content={self.raw_content}>'
    
    def __eq__(self, other):
        return self.type == other.type and self.raw_content == other.raw_content    

    def to_dict(self):
        if self.type == 'text':
            return {'type': 'text', 'text': self.raw_content}
        elif self.type == 'image_url':
            return {'type': 'image_url', 'image_url': {'url': self.raw_content, 'detail': 'low'}}
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(type=data['type'], raw_content=data[data['type']])
    
    @property
    def token_count(self) -> int:
        tokenizer = tiktoken.get_encoding('cl100k_base')
        if self.type == 'text':
            return len(tokenizer.encode(self.raw_content))
        elif self.type == 'image_url':
            return 85
        return 0

class BaseChatMessage:
    def __init__(self, 
                 role: Literal['user', 'assistant', 'system', 'tool'], 
                 content: str | list[MessageContentElement],
                 *,
                 name: str | discord.User | discord.Member | None = None,
                 timestamp: datetime | None = None,
                 token_count: int | None = None,
                 tool_call_id: str | None = None,
                 tool_calls: list[dict] | None = None):
        self.role = role
        
        self.__content = content
        self.__name = name
        self.__timestamp = timestamp or datetime.now(pytz.utc)
        self.__token_count = token_count
        
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls or []
        
        self._tool_used : str | None = None
        
    def __repr__(self):
        return f'<BaseChatMessage role={self.role} content={self.content} name={self.name}>'
    
    def __eq__(self, other):
        return self.role == other.role and self.content == other.content and self.name == other.name
    
    # Propriétés
    
    @property
    def content(self) -> list[MessageContentElement]:
        if isinstance(self.__content, str):
            return [MessageContentElement(type='text', raw_content=self.__content)]
        elif isinstance(self.__content, list):
            return self.__content
        return []
    
    @property
    def name(self) -> str | None:
        if isinstance(self.__name, str):
            return clean_name(self.__name)
        elif isinstance(self.__name, discord.User) or isinstance(self.__name, discord.Member):
            return clean_name(self.__name.name)
        return None
    
    @property
    def timestamp(self) -> datetime:
        return self.__timestamp
    
    @property
    def token_count(self) -> int:
        if self.__token_count is not None:
            return self.__token_count
        return sum([element.token_count for element in self.content])
    
    # Méthodes
    
    def to_dict(self) -> dict:
        if self.name:
            return {
                'role': self.role,
                'content': [element.to_dict() for element in self.content],
                'name': self.name
            }
        return {
            'role': self.role,
            'content': [element.to_dict() for element in self.content]
        }

class SystemChatMessage(BaseChatMessage):
    """Représente un message du système, qui n'est pas généré par un utilisateur ou un assistant"""
    def __init__(self, 
                 content: str):
        super().__init__(role='system', content=content, name=None, timestamp=None, token_count=None, tool_call_id=None)

class UserChatMessage(BaseChatMessage):
    """Repésente un message d'un utilisateur, avec un nom associé"""
    def __init__(self, 
                 content: str | list[MessageContentElement],
                 name: str | discord.User | discord.Member):
        super().__init__(role='user', content=content, name=name, timestamp=None, token_count=None, tool_call_id=None)

class AssistantChatMessage(BaseChatMessage):
    """Représente une réponse générée par l'IA"""
    def __init__(self, 
                 content: str | list[MessageContentElement],
                 token_count: int | None = None): # Le nb de tokens est renvoyé par l'API donc on peut le passer en paramètre
        super().__init__(role='assistant', content=content, name=None, timestamp=None, token_count=token_count, tool_call_id=None)

class AssistantToolCallChatMessage(BaseChatMessage):
    """Représente un message d'assistant qui appelle un outil"""
    def __init__(self,
                 tool_calls: list[dict]):
        super().__init__(role='assistant', content='', name=None, timestamp=None, token_count=None, tool_call_id=None, tool_calls=tool_calls)
        
    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'tool_calls': self.tool_calls
        }
    
class ToolChatMessage(BaseChatMessage):
    """Repésente un message d'outil, qui est une réponse à un appel d'outil"""
    def __init__(self,
                 content: str,
                 name: str,
                 tool_call_id: str):
        super().__init__(role='tool', content=content, name=name, timestamp=None, token_count=None, tool_call_id=tool_call_id)
        
    def to_dict(self) -> dict:
        return {
            'role': self.role,
            'content': str(self.content),
            'name': self.name,
            'tool_call_id': self.tool_call_id
        }

class ChatSession:
    def __init__(self,
                 cog: 'GPT',
                 channel: discord.abc.GuildChannel, 
                 system_prompt: str,
                 *,
                 temperature: float = 1.0,
                 max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                 max_context_tokens: int = MAX_CONTEXT_TOKENS):
        """Session de chat avec un modèle GPT4o-mini

        :param cog: Cog parent
        :param channel: Salon de discussion où se déroule la discussion
        :param system_prompt: Prompt initial pour le système 
        :param temperature: Créativité du modèle, entre 0 et 2, par défaut 1
        :param max_completion_tokens: Nombre maximum de tokens à générer
        :param max_context_tokens: Nombre maximum de tokens à garder en mémoire
        """
        self.__cog = cog
        self.channel = channel
        self.guild = channel.guild
        
        self._initial_system_prompt = system_prompt
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.max_context_tokens = max_context_tokens
        
        self.messages : list[BaseChatMessage] = []
        
        self._active_completion = False
        
    @property
    def system_prompt(self) -> BaseChatMessage:
        data = {
            'channel_name': self.channel.name,
            'guild_name': self.guild.name,
            'bot_name': self.guild.me.name,
            'system_prompt': self._initial_system_prompt,
            'current_date': datetime.now().strftime('%d/%m/%Y à %H:%M')
        }
        return SystemChatMessage(FULL_SYSTEM_PROMPT(data))
    
    # Contrôle de l'historique des messages
    
    async def resume(self, n: int = 10):
        """Récupère les X messages précédents du salon pour reconstituer l'historique"""
        if not isinstance(self.channel, (discord.TextChannel, discord.Thread)):
            return
        async for msg in self.channel.history(limit=n):
            if not msg.author == self.guild.me:
                elements = await self.__cog._extract_content_from_message(msg)
                self.messages.append(UserChatMessage(elements, msg.author))
            else:
                self.messages.append(AssistantChatMessage(msg.content))
    
    def add_message(self, message: BaseChatMessage):
        self.messages.append(message)
    
    def add_messages(self, messages: list[BaseChatMessage]):
        self.messages.extend(messages)
        
    def get_messages(self, cond: Callable[[BaseChatMessage], bool]):
        return [message for message in self.messages if cond(message)]
    
    def clear_messages(self, cond: Callable[[BaseChatMessage], bool]):
        self.messages = [message for message in self.messages if not cond(message)]
        
    def clear_all_messages(self):
        self.messages = []
        
    def get_context(self, include_system_prompt: bool = True):
        """Renvoie les X derniers messages dans la limite de max_context_tokens"""
        context = []
        token_count = self.system_prompt.token_count if include_system_prompt else 0
        for message in reversed(self.messages): # On inverse l'ordre pour avoir les messages les plus récents en premier
            # Les messages d'outils sont toujours inclus
            if message.role == 'tool' or message.tool_calls:
                context.append(message)
                continue
            token_count += message.token_count
            if token_count > self.max_context_tokens:
                break
            context.append(message)
        if include_system_prompt:
            context.append(self.system_prompt)
            
        # Nettoyage des messages trop anciens (1h avant le message le plus ancien du contexte)
        if context:
            oldest_message = context[-1]
            cutoff = oldest_message.timestamp - timedelta(hours=1)
            self.clear_messages(lambda message: message.timestamp < cutoff)
    
        return context[::-1] # On inverse l'ordre pour remettre les messages dans l'ordre chronologique
    
    # Interaction avec l'IA
    
    async def complete(self) -> AssistantChatMessage | ToolChatMessage | None:
        messages = [message.to_dict() for message in self.get_context()]
        
        counter = 0
        while self._active_completion:
            await asyncio.sleep(0.1)
            if counter > 100: # ~10s
                break
            counter += 1 # Garde fou au cas où l'IA ne répondrait pas à une requête précédente
        self._active_completion = True
        
        try:
            completion = await self.__cog.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                tools=GPT_TOOLS if ENABLE_TOOL_USE else [], # type: ignore
                tool_choice='auto',
                timeout=30
            )
        except Exception as e:
            if 'invalid_image_url' in str(e): 
                logger.error(f'Error while generating completion: {e}\n-- Deleting messages with images...')
                self.clear_messages(lambda message: any([element.type == 'image_url' for element in message.content]))
                return await self.complete()
            logger.error(f'Error while generating completion: {e}')
            raise e
        finally:
            self._active_completion = False
            
        if not completion.choices:
            return None
        
        message = completion.choices[0].message
        content = message.content if message.content else ''
        usage = completion.usage.total_tokens if completion.usage else 0
        tool_used = None
        
        if ENABLE_TOOL_USE:
            tool_msg = None
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                call_data = {
                    'id': tool_call.id,
                    'type': 'function',
                    'function': {
                        'arguments': tool_call.function.arguments,
                        'name': tool_call.function.name
                    }
                }
                tool_used = tool_call.function.name
                calling_msg = AssistantToolCallChatMessage([call_data])
                
                if tool_call.function.name == 'get_user_info':
                    user_name = json.loads(tool_call.function.arguments)['user']
                    key = json.loads(tool_call.function.arguments)['key']
                    user_id = self.__cog.fetch_user_id_from_name(self.guild, user_name)
                    if user_id:
                        notes = self.__cog.get_user_info(user_id, key)
                        if not notes:
                            tool_msg = ToolChatMessage(json.dumps({'user': user_name, 'key': key, 'value': 'Aucune note trouvée'}), tool_call.function.name, tool_call.id)
                        else:
                            tool_msg = ToolChatMessage(json.dumps({'user': user_name, 'key': key, 'value': notes}), tool_call.function.name, tool_call.id)
                elif tool_call.function.name == 'set_user_info':
                    arguments = json.loads(tool_call.function.arguments)
                    user_name = arguments['user']
                    key = arguments['key']  
                    value = arguments['value']
                    user_id = self.__cog.fetch_user_id_from_name(self.guild, user_name)
                    if not user_id:
                        tool_msg = ToolChatMessage(json.dumps({'user': user_name, 'key': key, 'value': 'Utilisateur non existant'}), tool_call.function.name, tool_call.id)
                    else:
                        self.__cog.set_user_info(user_id, key, value)
                        tool_msg = ToolChatMessage(json.dumps({'user': user_name, 'key': key, 'value': value}), tool_call.function.name, tool_call.id)
                elif tool_call.function.name == 'get_all_user_info':
                    user_name = json.loads(tool_call.function.arguments)['user']
                    user_id = self.__cog.fetch_user_id_from_name(self.guild, user_name)
                    if not user_id:
                        tool_msg = ToolChatMessage(json.dumps({'user': user_name, 'value': 'Utilisateur non existant'}), tool_call.function.name, tool_call.id)
                    else:
                        notes = self.__cog.get_all_user_info(user_id)
                        if not notes:
                            tool_msg = ToolChatMessage(json.dumps({'user': user_name, 'value': 'Aucune note trouvée sur cet utilisateur'}), tool_call.function.name, tool_call.id)
                        else:
                            tool_msg = ToolChatMessage(json.dumps({'user': user_name, 'value': notes}), tool_call.function.name, tool_call.id)
                    
                if tool_msg:
                    self.add_messages([calling_msg, tool_msg])
                    return await self.complete()
        
        answer_msg = AssistantChatMessage(content, token_count=usage)
        answer_msg._tool_used = tool_used
        return answer_msg


class GPT(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # Configuration des salons
        chatconfig = dataio.TableBuilder(
            '''CREATE TABLE IF NOT EXISTS chatconfig (
                channel_id INTEGER PRIMARY KEY,
                system_prompt TEXT,
                temperature REAL DEFAULT 1.0
                )'''
        )
        self.data.link(discord.Guild, chatconfig)
        
        # Table pour stocker les notes de l'assistant sur les utilisateurs
        memory = dataio.TableBuilder(
            '''CREATE TABLE IF NOT EXISTS memory (
                user_id INTEGER,
                key TEXT,
                value TEXT,
                PRIMARY KEY (user_id, key)
                )'''
        )
        self.data.link('global', memory)
        
        self.client = AsyncOpenAI(
                api_key=self.bot.config['OPENAI_API_KEY'], # type: ignore
            )
        
        self.create_audio_transcription = app_commands.ContextMenu(
            name="Transcription audio",
            callback=self.transcript_audio_callback)
        self.bot.tree.add_command(self.create_audio_transcription)
        
        self._sessions : dict[int, ChatSession] = {}
        
    async def cog_unload(self):
        self.data.close_all()
        await self.client.close()
        
    # Session de chat -----------------------------------------------------------
        
    async def _get_session(self, channel: discord.abc.GuildChannel) -> ChatSession: 
        if channel.id in self._sessions:
            return self._sessions[channel.id]
        
        config = self.data.get(channel.guild).fetchone('SELECT * FROM chatconfig WHERE channel_id = ?', channel.id)
        if not config:
            # On crée une nouvelle entrée pour ce salon avec les valeurs par défaut
            config = self.data.get(channel.guild).fetchone('INSERT INTO chatconfig(channel_id, system_prompt) VALUES (?, ?)', channel.id, DEFAULT_SYSTEM_PROMPT)
            self.data.get(channel.guild).commit()
            if not config:
                raise ValueError('Could not create chatconfig entry')
        
        session = ChatSession(self,
                              channel, 
                              system_prompt=config['system_prompt'],
                              temperature=config['temperature'])
        self._sessions[channel.id] = session
        if isinstance(channel, (discord.TextChannel, discord.Thread)):
            await session.resume()
        return session
    
    def _clear_session(self, channel: discord.abc.GuildChannel):
        self._sessions.pop(channel.id, None)
        
    async def _extract_content_from_message(self, message: discord.Message) -> list[MessageContentElement]:
        message_content = []
        guild = message.guild
        if not guild:
            return message_content
        
        ref_message = None
        if message.reference and message.reference.message_id:
            ref_message = await message.channel.fetch_message(message.reference.message_id)
        
        if message.content:
            content = message.content.replace(guild.me.mention, '').strip()
            author_name = message.author.name if not message.author.bot else f'{message.author.name} (bot)'
            if ref_message and ref_message.content:
                ref_name = ref_message.author.name if not ref_message.author.bot else f'{ref_message.author.name} (bot)'
                content = f'[Message cité] {ref_name}: {ref_message.content}\n[En réponse à {ref_name}] {author_name}: {content}'
            else:
                content = f'{author_name}: {content}'
            message_content.append(MessageContentElement('text', content))
        else:
            # On regarde si y'a pas de texte dans un embed
            for embed in message.embeds:
                if embed.description:
                    message_content.append(MessageContentElement('text', embed.description))
        
        image_urls = []
        for msg in [message, ref_message]:
            if not msg:
                continue
            # Pièces jointes
            for attachment in msg.attachments:
                if attachment.content_type and attachment.content_type.startswith('image'):
                    image_urls.append(attachment.url)
            # Dans le texte d'un message
            for match in re.finditer(r'(https?://[^\s]+)', msg.content):
                url = match.group(0)
                if url.endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
                    image_urls.append(url)
            # Dans le contenu d'un embed
            for embed in msg.embeds:
                if embed.image:
                    image_urls.append(embed.image.url)
        if image_urls:
            message_content.extend([MessageContentElement('image_url', url) for url in image_urls])
        
        return message_content
    
    # Notes de l'assistant ------------------------------------------------------
    
    def fetch_user_id_from_name(self, guild: discord.Guild, name: str) -> int | None:
        user = discord.utils.find(lambda u: u.name == name, guild.members)
        return user.id if user else None
    
    def get_user_info(self, user: discord.User | discord.Member | int, key: str) -> str | None:
        user_id = user.id if isinstance(user, (discord.User, discord.Member)) else user
        notes = self.data.get('global').fetchone('SELECT value FROM memory WHERE user_id = ? AND key = ?', user_id, key)
        if not notes:
            # On cherche la note la plus proche
            keys = self.data.get('global').fetchall('SELECT key FROM memory WHERE user_id = ?', user_id)
            closest_key = fuzzy.extract_one(key, [k['key'] for k in keys])
            if closest_key:
                notes = self.data.get('global').fetchone('SELECT value FROM memory WHERE user_id = ? AND key = ?', user_id, closest_key[0])
        return notes['value'] if notes else None
    
    def get_all_user_info(self, user: discord.User | discord.Member | int) -> dict:
        user_id = user.id if isinstance(user, (discord.User, discord.Member)) else user
        notes = self.data.get('global').fetchall('SELECT key, value FROM memory WHERE user_id = ?', user_id)
        return {note['key']: note['value'] for note in notes}
    
    def set_user_info(self, user: discord.User | discord.Member | int, key: str, value: str):
        user_id = user.id if isinstance(user, (discord.User, discord.Member)) else user
        self.data.get('global').execute('INSERT OR REPLACE INTO memory(user_id, key, value) VALUES (?, ?, ?)', user_id, key, value)
        
    def delete_user_info(self, user: discord.User | discord.Member | int, key: str):
        user_id = user.id if isinstance(user, (discord.User, discord.Member)) else user
        self.data.get('global').execute('DELETE FROM memory WHERE user_id = ? AND key = ?', user_id, key)
        
    def clear_user_info(self, user: discord.User | discord.Member | int):
        user_id = user.id if isinstance(user, (discord.User, discord.Member)) else user
        self.data.get('global').execute('DELETE FROM memory WHERE user_id = ?', user_id)
    
    # Audio --------------------------------------------------------------------
    
    async def get_audio(self, message: discord.Message) -> io.BytesIO | None:
        """Récupère le fichier audio d'un message."""
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('audio'):
                buffer = io.BytesIO()
                buffer.name = attachment.filename
                await attachment.save(buffer, seek_begin=True)
                return buffer
        return None
    
    async def get_audio_from_video(self, message: discord.Message) -> Path | None:
        """Récupère le fichier audio d'une vidéo."""
        for attachment in message.attachments:
            if attachment.content_type and attachment.content_type.startswith('video'):
                path = self.data.get_subfolder('temp', create=True) / f'{datetime.now().timestamp()}.mp4'
                await attachment.save(path)
                clip = VideoFileClip(str(path))
                audio = clip.audio
                if not audio:
                    return None
                audio_path = path.with_suffix('.wav')
                audio.write_audiofile(str(audio_path))
                clip.close()
                
                os.remove(str(path))
                return audio_path
        return None
    
    async def audio_transcription(self, file: io.BytesIO | Path, model: str = 'whisper-1'):
        try:
            transcript = await self.client.audio.transcriptions.create(
                model=model,
                file=file
            )
        except Exception as e:
            logger.error(f"ERREUR OpenAI : {e}", exc_info=True)
            return None
        if isinstance(file, io.BytesIO):
            file.close()
        return transcript.text
    
    async def handle_audio_transcription(self, interaction: Interaction, message: discord.Message, transcript_author: discord.Member | discord.User):
        await interaction.response.defer()
        
        attach_type = message.attachments[0].content_type
        if not message.attachments or not attach_type:
            return await interaction.followup.send(f"**Erreur Discord** × Aucun fichier supporté n'est attaché au message.", ephemeral=True)
        
        notif = await interaction.followup.send(f"Récupération de l'audio en cours...", ephemeral=True, wait=True)
        
        if attach_type.startswith('video'):
            await notif.edit(content="Conversion de la vidéo en fichier audio...")
            file_or_buffer = await self.get_audio_from_video(message)
        elif attach_type.startswith('audio'):
            file_or_buffer = await self.get_audio(message)
        else:
            return await interaction.followup.send(f"**Erreur interne** × Le fichier audio n'a pas pu être récupéré.", ephemeral=True)
    
        if not file_or_buffer:
            return await interaction.followup.send(f"**Erreur interne** × Le fichier audio n'a pas pu être récupéré.", ephemeral=True)
        
        await notif.edit(content=f"Transcription en cours de traitement...")
        
        transcript = await self.audio_transcription(file_or_buffer)
        if not transcript:
            return await notif.edit(content=f"**Erreur OpenAI** × Aucun texte n'a pu être extrait de ce fichier.")
        
        await interaction.delete_original_response()
        if type(file_or_buffer) == Path:
            os.remove(str(file_or_buffer))

        if len(transcript) > 1950:
            return await message.reply(f"**Transcription demandée par {transcript_author.mention}** :\n>>> {transcript[:1950]}...", mention_author=False)
        
        await message.reply(f"**Transcription demandée par {transcript_author.mention}** :\n>>> {transcript}", mention_author=False)
    
    async def transcript_audio_callback(self, interaction: Interaction, message: discord.Message):
        """Callback pour demander la transcription d'un message audio via le menu contextuel."""
        if interaction.channel_id != message.channel.id:
            return await interaction.response.send_message("**Action impossible** × Le message doit être dans le même salon", ephemeral=True)
        if not message.attachments:
            return await interaction.response.send_message("**Erreur** × Aucun fichier n'est attaché au message.", ephemeral=True)
        return await self.handle_audio_transcription(interaction, message, interaction.user)
    
    # Listener -----------------------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not message.guild:
            return # On ne gère pas les messages privés
        if not isinstance(message.channel, discord.abc.GuildChannel):
            return
        if not message.channel.permissions_for(message.guild.me).send_messages:
            return
        if message.mention_everyone:
            return
        
        guild = message.guild
        session = await self._get_session(message.channel)
        if not session:
            return
        
        message_content = await self._extract_content_from_message(message)
            
        # Si le message est un message audio et que le bot est mentionné, on le transcrit automatiquement
        if message.attachments and message.attachments[0].content_type and message.attachments[0].content_type.startswith('audio'):
            if message.author.bot:
                return # On ne transcrit pas les messages audio des autres bots
            if guild.me.mentioned_in(message):
                audio = await self.get_audio(message)
                if audio:
                    transcript = await self.audio_transcription(audio)
                    if transcript:
                        message_content.append(MessageContentElement('text', f"[Transcription du message audio] {transcript}"))
        
        session.add_message(UserChatMessage(message_content, message.author))
        
        if message.author.bot:
            return # On ne répond pas aux autres bots
        
        if guild.me.mentioned_in(message):
            async with message.channel.typing():
                completion = await session.complete()
                if not completion:
                    return await message.reply("**Erreur** × Je n'ai pas pu générer de réponse.\n-# Réessayez dans quelques instants. Si le problème persiste, demandez à un modérateur de faire `/resethistory`.", mention_author=False)
                session.add_message(completion)
                
                # Ajout d'un emoji si un outil a été utilisé (on a noté le message d'outil juste avant)
                if completion._tool_used:
                    try:
                        if completion._tool_used in ['get_user_info', 'get_all_user_info']:
                            await message.add_reaction('🔍')
                        elif completion._tool_used == 'set_user_info':
                            await message.add_reaction('✏️')
                    except discord.HTTPException:
                        pass
                
                if not completion.content or not completion.content[0].raw_content:
                    return await message.reply("**Erreur** × Je n'ai pas pu générer de réponse.\n-# Réessayez dans quelques instants. Si le problème persiste, demandez à un modérateur de faire `/resethistory`.", mention_author=False)
                await message.reply(completion.content[0].raw_content, mention_author=False, suppress_embeds=True, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False, replied_user=True))
                
                
    # COMMANDES =================================================================
    
    @app_commands.command(name='systemprompt')
    @app_commands.guild_only()
    @app_commands.rename(new_prompt='instructions')
    async def cmd_systemprompt(self, interaction: Interaction, new_prompt: str | None = None):
        """Consulter et modifier les instructions système de l'assistant.
        
        :param new_prompt: Nouvelles instructions système, entre 10 et 500 caractères"""
        if not isinstance(interaction.channel, discord.abc.GuildChannel):
            return await interaction.response.send_message("**Action impossible** × Cette commande ne fonctionne pas en message privé.", ephemeral=True)
        
        session = await self._get_session(interaction.channel)
        if not session:
            return await interaction.response.send_message("**Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        if new_prompt is None:
            prompt = SystemPromptModal(session._initial_system_prompt)
            await interaction.response.send_modal(prompt)
            if await prompt.wait(): # Si True, timeout, sinon on a reçu une réponse
                return await interaction.response.send_message("**Action annulée** × Vous avez pris trop de temps pour répondre.", ephemeral=True)
            new_prompt = prompt.new_system_prompt.value
        elif 500 < len(new_prompt) < 10:
            return await interaction.response.send_message("**Erreur** × Les instructions doivent contenir entre 10 et 500 caractères.", ephemeral=True)
            
        self.data.get(interaction.channel.guild).execute('UPDATE chatconfig SET system_prompt = ? WHERE channel_id = ?', new_prompt, interaction.channel.id)
        
        session._initial_system_prompt = new_prompt
        session.clear_all_messages()
        await interaction.followup.send(f"**Instructions système mises à jour** · Voici les nouvelles instructions :\n> *{new_prompt}*\n-# Afin d'éviter que les nouvelles instructions rentrent en conflit avec les précédents messages de l'assistant (sous d'anciennes instructions), sa mémoire a été réinitalisée.", ephemeral=True)
        
    @app_commands.command(name='temperature')
    @app_commands.guild_only()
    @app_commands.rename(temp='température')
    async def cmd_temperature(self, interaction: Interaction, temp: app_commands.Range[float, 0.0, 2.0]):
        """Modifier le degré de créativité de l'assistant.

        :param temp: Température de génération, entre 0.0 et 2.0"""
        if not isinstance(interaction.channel, discord.abc.GuildChannel):
            return await interaction.response.send_message("**Action impossible** × Cette commande ne fonctionne pas en message privé.", ephemeral=True)
        
        session = await self._get_session(interaction.channel)
        if not session:
            return await interaction.response.send_message("**Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        # On met à jour la température
        self.data.get(interaction.channel.guild).execute('UPDATE chatconfig SET temperature = ? WHERE channel_id = ?', temp, interaction.channel.id)
        session.temperature = temp
        if temp >= 1.5:
            return await interaction.response.send_message(f"**Température mise à jour** · La température de génération est désormais de ***{temp}***.\n-# Attention, une température élevée peut entraîner des réponses incohérentes.", ephemeral=True)
        await interaction.response.send_message(f"**Température mise à jour** · La température de génération est désormais de {temp}.", ephemeral=True)
    
    @app_commands.command(name='info')
    @app_commands.guild_only()
    async def cmd_info(self, interaction: Interaction):
        """Afficher les informations sur l'assistant sur la session en cours."""
        if not isinstance(interaction.channel, discord.abc.GuildChannel):
            return await interaction.response.send_message("**Action impossible** × Cette commande ne fonctionne pas en message privé.", ephemeral=True)
        
        embed = discord.Embed(title="Informations sur l'assitant", color=discord.Color(0x000001))
        embed.set_thumbnail(url=interaction.channel.guild.me.display_avatar.url)
        embed.set_footer(text="Implémentation de GPT4o-mini et Whisper-1 (par OpenAI)", icon_url="https://static-00.iconduck.com/assets.00/openai-icon-2021x2048-4rpe5x7n.png")
        session = await self._get_session(interaction.channel)
        if not session:
            return await interaction.response.send_message("**Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        # Informations sur l'assistant
        embed.add_field(name="Instructions système", value=pretty.codeblock(session._initial_system_prompt), inline=False)
        embed.add_field(name="Température de génération", value=pretty.codeblock(str(session.temperature), lang='css'))
        
        # Informations sur la session
        embed.add_field(name="Messages en mémoire", value=pretty.codeblock(str(len(session.messages))))
        embed.add_field(name="Tokens en mémoire", value=pretty.codeblock(str(sum([message.token_count for message in session.messages]))))
        
        await interaction.response.send_message(embed=embed)
    
    @app_commands.command(name='resethistory')
    @app_commands.guild_only()
    @app_commands.default_permissions(manage_messages=True)
    async def cmd_resethistory(self, interaction: Interaction):
        """Réinitialiser les messages en mémoire de l'assistant."""
        if not isinstance(interaction.channel, discord.abc.GuildChannel):
            return await interaction.response.send_message("**Action impossible** × Cette commande ne fonctionne pas en message privé.", ephemeral=True)
        
        session = await self._get_session(interaction.channel)
        if not session:
            return await interaction.response.send_message("**Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        session.clear_all_messages()
        await interaction.response.send_message("**Mémoire réinitialisée** · L'historique des messages a été effacé.", ephemeral=True)
        
    memory_group = app_commands.Group(name='memory', description="Gestion des notes internes de l'assistant")

    @memory_group.command(name='show')
    async def cmd_show_memory(self, interaction: Interaction):
        """Consulter les notes de l'assistant associées à vous."""
        user = interaction.user
        notes = self.get_all_user_info(user)
        if not notes:
            return await interaction.response.send_message(f"**Notes de l'assistant** · Aucune note n'est associée à vous.", ephemeral=True)
        
        text = '\n'.join([f"{key} : {value}" for key, value in notes.items()])
        embed = discord.Embed(title=f"Notes de l'assistant [BETA]", description=pretty.codeblock(text), color=discord.Color(0x000001))
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.set_footer(text="Ces notes sont stockées dans une base de données locale.")
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    @memory_group.command(name='delete')
    @app_commands.rename(key='clé')
    async def cmd_delete_memory(self, interaction: Interaction, key: str | None = None):
        """Supprimer les notes de l'assistant associées à vous.
        
        :param key: Clé de la note à supprimer"""
        user = interaction.user
        if not key:
            # On demande confirmation
            view = ConfirmView(user)
            await interaction.response.send_message(f"**Confirmation** · Êtes-vous sûr de vouloir supprimer toutes les notes de l'assistant associées à vous ?", ephemeral=True, view=view)
            await view.wait()
            if not view.value:
                return await interaction.edit_original_response(content="**Action annulée** · Les notes de l'assistant n'ont pas été supprimées.", view=None)
            self.clear_user_info(user)
            return await interaction.edit_original_response(content="**Notes supprimées** · Toutes les notes de l'assistant associées à vous ont été supprimées.", view=None)
        
        notes = self.get_user_info(user, key)
        if not notes:
            return await interaction.response.send_message(f"**Notes de l'assistant** · Aucune note n'est associée à vous pour la clé `{key}`.", ephemeral=True)
        
        # On demande confirmation
        view = ConfirmView(user)
        await interaction.response.send_message(f"**Confirmation** · Êtes-vous sûr de vouloir supprimer la note de l'assistant associée à vous pour la clé `{key}` ?", ephemeral=True, view=view)
        await view.wait()
        if not view.value:
            return await interaction.edit_original_response(content="**Action annulée** · La note de l'assistant n'a pas été supprimée.", view=None)
        
        self.delete_user_info(user, key)
        await interaction.edit_original_response(content=f"**Note supprimée** · La note de l'assistant associée à vous pour la clé `{key}` a été supprimée.", view=None)

    @cmd_delete_memory.autocomplete('key')
    async def autocomplete_key_callback(self, interaction: Interaction, current: str):
        user = interaction.user 
        keys = self.get_all_user_info(user).keys()
        fuzz = fuzzy.finder(current, keys)
        return [app_commands.Choice(name=key, value=key) for key in fuzz][:10]

async def setup(bot):
    await bot.add_cog(GPT(bot))
