import io
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import discord
import pytz
import tiktoken
import unidecode
from discord import Interaction, app_commands
from discord.ext import commands
from googlesearch import search
from moviepy.editor import VideoFileClip
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message_tool_call import \
    ChatCompletionMessageToolCall

from common import dataio
from common.utils import fuzzy, pretty

logger = logging.getLogger(f'MARIA.{__name__.split(".")[-1]}')

# Prompt système complété du fonctionnement interne de l'assistant
META_SYSTEM_PROMPT = lambda d: f"""[FONCTIONNEMENT]
Tu es {d['assistant_name']}. Tu réponds aux utilisateurs d'un salon de discussion.
Les messages des utilisateurs sont précédés de leurs noms. Ne met pas le tien devant tes réponses.
Tu dois suivre scrupuleusement les [INSTRUCTIONS] données ci-après.
[INFORMATIONS]
- Serveur : {d['guild_name']}
- Date/heure : {d['current_time']}
[OUTILS]
- Notes d'utilisateurs : utilise-les dès que nécessaire pour récupérer, rechercher ou mettre à jour des informations sur les utilisateurs.
- Recherche web : pour récupérer des liens vers des pages web.
- Emojis du serveur : pour récupérer la liste des emojis Discord du serveur.
[INSTRUCTIONS]
{d['system_prompt']}
"""

# Constantes
DEFAULT_SYSTEM_PROMPT = "Tu es un assistant utile et familier qui répond aux questions des différents utilisateurs de manière concise et simple."
MAX_COMPLETION_TOKENS = 500 # Nombre maximal de tokens pour une complétion
CONTEXT_WINDOW = 10000 # Nombre de tokens à conserver dans le contexte de conversation
CONTEXT_MAX_AGE = timedelta(hours=48) # Durée maximale de conservation des messages dans le contexte de conversation
CONTEXT_CLEANUP_INTERVAL = timedelta(hours=1) # Intervalle de nettoyage du contexte de conversation
VISION_DETAIL = 'low' # Détail de la vision artificielle
ENABLE_TOOLS = True # Activation des outils de l'assistant

# Définition des outils de l'assistant
GPT_TOOLS = [
    { # Récupération des informations d'un utilisateur
        'type': 'function',
        'function': {
            'name': 'get_user_info',
            'description': "Renvoie une clé-valeur d'information sur un utilisateur. Renvoie l'ensemble des informations si aucune clé n'est renseignée.",  
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['user', 'key'],
                'properties': {
                    'user': {'type': 'string', 'description': "Nom de l'utilisateur à rechercher."},
                    'key': {'type': ['string', 'null'], 'description': "Clé de l'information à récupérer (ex. 'age', 'ville' etc.). Ne pas renseigner pour récupérer l'ensemble des notes en mémoire."}
                },
                'additionalProperties': False
            }
        }
    },
    { # Recherche d'utilisateurs par clé
        'type': 'function',
        'function': {
            'name': 'find_users_by_key',
            'description': "Recherche les utilisateurs selon une clé d'information. Renvoie un dictionnaire avec les utilisateurs trouvés et leurs notes.",
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['key'],
                'properties': {
                    'key': {'type': 'string', 'description': "Clé de l'information à rechercher (ex. 'age')."}
                },
                'additionalProperties': False
            }
        }
    },
    { # Mise à jour des informations d'un utilisateur
        'type': 'function',
        'function': {
            'name': 'set_user_info',
            'description': "Met à jour la note liée à une clé pour un utilisateur.",
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['user', 'key', 'value'],
                'properties': {
                    'user': {'type': 'string', 'description': "Nom de l'utilisateur à mettre à jour."},
                    'key': {'type': 'string', 'description': "Clé de l'information à mettre à jour."},
                    'value': {'type': ['string', 'null'], 'description': "Valeur à mettre à jour. Ne pas renseigner pour supprimer la clé."}
                },
                'additionalProperties': False
            }
        }
    },
    { # Recherche de pages web
     'type': 'function',
        'function': {
            'name': 'search_web_pages',
            'description': "Recherche des liens vers des pages web. Renvoie les titres, descriptions et URLs des résultats.",
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': ['query', 'lang', 'num_results'],
                'properties': {
                    'query': {'type': 'string', 'description': "Requête de recherche."},
                    'lang': {'type': 'string', 'description': "Langue de recherche (ex. 'fr')."},
                    'num_results': {'type': 'integer', 'description': "Nombre de résultats à renvoyer. (Max. 10)."}
                },
                'additionalProperties': False
            }
        }
    },
    { # Récupération des emojis du serveur
     'type': 'function',
        'function': {
            'name': 'get_server_emojis',
            'description': "Récupère la liste des emojis Discord du serveur.",
            'strict': True,
            'parameters': {
                'type': 'object',
                'required': [],
                'properties': {},
                'additionalProperties': False
            }
        }   
    }
]

# Utils ----------------------------------------------------------------------

def sanitize_text(text: str) -> str:
    """Retire les caractères spéciaux d'un texte."""
    text = ''.join([c for c in unidecode.unidecode(text) if c.isalnum() or c.isspace()]).rstrip()
    return re.sub(r"[^a-zA-Z0-9_-]", "", text[:32])

# UI -------------------------------------------------------------------------

class SystemPromptModal(discord.ui.Modal, title="Modifier les instructions"):
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

# CLASSES ======================================================================

class MessageElement:
    """Représente un élément de message dans un contexte de conversation."""
    def __init__(self,
                 type: Literal['text', 'image_url'],
                 raw_data: str) -> None:
        self.type = type
        self.raw_data = raw_data
        
    def __repr__(self) -> str:
        return f'<MessageElement type={self.type} raw_data={self.raw_data}>'
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, MessageElement):
            return False
        return self.type == value.type and self.raw_data == value.raw_data
    
    @property
    def token_count(self) -> int:
        tokenizer = tiktoken.get_encoding('cl100k_base')
        if self.type == 'text':
            return len(tokenizer.encode(self.raw_data))
        elif self.type == 'image_url':
            return 85
        return 0
        
    def to_dict(self) -> dict:
        if self.type == 'text':
            return {'type': 'text', 'text': self.raw_data}  
        elif self.type == 'image_url':
            return {'type': 'image_url', 'image_url': {'url': self.raw_data, 'detail': VISION_DETAIL}}
        return {}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MessageElement':
        if data['type'] == 'text':
            return cls('text', data['text'])
        elif data['type'] == 'image_url':
            return cls('image_url', data['image_url']['url'])
        raise ValueError('Invalid data')


class ContextMessage:
    """Repésente un message dans la conversation."""
    def __init__(self,
                 role: Literal['user', 'assistant', 'system', 'tool'],
                 content: str | Iterable[MessageElement] | None,
                 *,
                 name: str | None = None,
                 timestamp: datetime = datetime.now(pytz.utc),
                 **kwargs) -> None:
        self.role = role
        self.name = sanitize_text(name) if name else None
        self.timestamp = timestamp
        
        self._raw_content = content
        self.__token_count : int = kwargs.get('token_count', 0)
    
    def __repr__(self) -> str:
        return f'<ContextMessage role={self.role} name={self.name} content={self.content}>'
    
    # Propriétés
    
    @property
    def content(self) -> list[MessageElement]:
        if isinstance(self._raw_content, str):
            return [MessageElement('text', self._raw_content)]
        elif isinstance(self._raw_content, list):
            return self._raw_content
        return []
        
    @property
    def token_count(self) -> int:
        if self.__token_count:
            return self.__token_count
        return sum([m.token_count for m in self.content])
    
    @property
    def payload(self) -> dict:
        if self.name:
            return {'role': self.role, 'content': [m.to_dict() for m in self.content], 'name': self.name}
        return {'role': self.role, 'content': [m.to_dict() for m in self.content]}
    
class SystemCtxMessage(ContextMessage):
    """Représente un message système dans la conversation."""
    def __init__(self,
                 content: str) -> None:
        super().__init__('system', content)
    
class AssistantCtxMessage(ContextMessage):
    """Représente un message de l'assistant dans la conversation."""
    def __init__(self,
                 content: str,
                 *,
                 timestamp: datetime = datetime.now(pytz.utc),
                 token_count: int = 0,
                 files: list[discord.File] = [],
                 tools_used: list[str] = [],
                 finish_reason: str | None = None) -> None:
        super().__init__('assistant', content, timestamp=timestamp, token_count=token_count)
        self._raw_content : str = content
        self.files = files
        self.tools_used = tools_used
        self.finish_reason : str | None = finish_reason
        
    @classmethod
    def from_gpt_payload(cls, payload: ChatCompletion) -> 'AssistantCtxMessage':
        if not payload.choices:
            raise ValueError('No completion choices found.')
        completion = payload.choices[0]
        finish_reason = completion.finish_reason
        usage = payload.usage.total_tokens if payload.usage else 0
        content = completion.message.content
        return cls(content if content else '...',
                   token_count=usage,
                   finish_reason=finish_reason)
        
class AssistantToolCall(ContextMessage):
    """Représente un appel d'outil dans la conversation."""
    def __init__(self,
                 tool_call: dict):
        super().__init__('assistant', None)
        self.tool_call = tool_call
        
    @property
    def payload(self) -> dict:
        return {'role': self.role, 'tool_calls': [self.tool_call]}
        
    @property
    def function_name(self) -> str:
        return self.tool_call['function']['name']
    
    @property
    def function_arguments(self) -> dict:
        return json.loads(self.tool_call['function']['arguments'])
    
    @classmethod
    def from_gpt_tool_call(cls, tool_call: ChatCompletionMessageToolCall) -> 'AssistantToolCall':
        return cls({
            'id': tool_call.id,
            'type': 'function',
            'function': {
                'name': tool_call.function.name,
                'arguments': tool_call.function.arguments
            }
        })
        
class ToolCtxMessage(ContextMessage):
    """Représente le retour de l'appel d'un outil dans la conversation."""
    def __init__(self,
                 content: dict,
                 tool_call_id: str) -> None:
        super().__init__('tool', json.dumps(content))
        self.tool_call_id = tool_call_id
        
        self.attachments = []
        
    @property
    def payload(self) -> dict:
        return {'role': self.role, 'content': self._raw_content, 'tool_call_id': self.tool_call_id}
    
class UserCtxMessage(ContextMessage):
    """Représente un message de l'utilisateur dans la conversation."""
    def __init__(self,
                 content: str | Iterable[MessageElement],
                 name: str | None = None,
                 *,
                 timestamp: datetime = datetime.now(pytz.utc),
                 channel: discord.TextChannel | discord.Thread | None = None) -> None:
        super().__init__('user', content, name=name, timestamp=timestamp)
        
        self._channel: discord.TextChannel | discord.Thread | None = channel

    @classmethod
    async def from_message(cls, message: discord.Message) -> 'UserCtxMessage':
        content = []
        guild = message.guild
        if not guild:
            raise ValueError('Guild not found.')
        
        ref_message = message.reference.resolved if message.reference else None
        if message.content:
            author_name = message.author.name if not message.author.bot else f'{message.author.name}[bot]'
            san_content = message.content.replace(guild.me.mention, '').strip()
            msg_content = f'{author_name}:{san_content}'
            if isinstance(ref_message, discord.Message) and ref_message.content:
                if not ref_message.author.bot: # On ne cite pas les messages de bots
                    ref_author_name = ref_message.author.name if not ref_message.author.bot else f'{ref_message.author.name}[bot]'
                    msg_content = f'[Début de citation] {ref_author_name}:{ref_message.clean_content} [Fin de citation]\n{msg_content}'
            content.append(MessageElement('text', msg_content))
                
        image_urls = []
        for msg in [message, ref_message]:
            if not isinstance(msg, discord.Message):
                continue
            
            for embed in msg.embeds:
                if embed.description:
                    if msg == message:
                        content.append(MessageElement('text', f'[Embed] {embed.description}'))
                    else:
                        content.append(MessageElement('text', f'[Début de citation] [Embed] {embed.description} [Fin de citation]'))
            
            for attachment in msg.attachments:
                if attachment.content_type and attachment.content_type.startswith('image'):
                    image_urls.append(attachment.url)
            for match in re.finditer(r'(https?://[^\s]+)', msg.content):
                url = match.group(1)
                url = re.sub(r'\?.*$', '', url)
                if url.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    image_urls.append(url)
            for embed in msg.embeds:
                if embed.image and embed.image.url:
                    image_urls.append(embed.image.url)
            
        if image_urls:
            content.extend([MessageElement('image_url', url) for url in image_urls])
            
        channel = message.channel if isinstance(message.channel, (discord.TextChannel, discord.Thread)) else None
            
        return cls(content, name=message.author.name, timestamp=message.created_at, channel=channel)
    
    
class ChatInteraction:
    """Représente un groupe de messages (questions, réponses et outils) dans une conversation."""
    def __init__(self, messages: list[ContextMessage]) -> None:
        self._messages = messages
        
    def __repr__(self) -> str:
        return f'<ChatInteraction messages={len(self.messages)}>'
    
    def __iter__(self) -> Iterable[ContextMessage]:
        return iter(self.messages)
    
    # Propriétés globales
    
    @property
    def id(self) -> str:
        """Identifiant unique de l'interaction"""
        return self._messages[0].timestamp.strftime('%Y%m%d%H%M%S%f')
    
    @property
    def total_token_count(self) -> int:
        return sum([m.token_count for m in self.messages])
    
    @property
    def completed(self) -> bool:
        """Renvoie True si l'interaction est complète (question et réponse)."""
        return len(self.messages) >= 2 and self.messages[-1].role == 'assistant'
    
    @property
    def contains_tool(self) -> bool:
        """Renvoie True si l'interaction contient un appel d'outil."""
        return any([m.role == 'tool' for m in self.messages])
    
    @property
    def contains_image(self) -> bool:
        """Renvoie True si un élément de l'interaction est une image."""
        return any([m.type == 'image_url' for elem in self.messages for m in elem.content])
    
    @property
    def retrieved_channel(self) -> discord.TextChannel | discord.Thread | None:
        """Tente de récupérer le salon de la dernière question de l'interaction."""
        for message in self.messages:
            if isinstance(message, UserCtxMessage):
                return message._channel
        return None
    
    # Propriétés des messages
    
    @property
    def messages(self) -> list[ContextMessage]:
        return self._messages
    
    @property
    def last_user_message(self) -> UserCtxMessage | None:
        for message in self.messages:
            if isinstance(message, UserCtxMessage):
                return message
        return None
    
    @property
    def last_assistant_message(self) -> AssistantCtxMessage | None:
        for message in self.messages:
            if isinstance(message, AssistantCtxMessage):
                return message
        return None
    
    # Gestion des messages
    
    def get_messages(self, role: Literal['user', 'assistant', 'system', 'tool']) -> list[ContextMessage]:
        return [m for m in self.messages if m.role == role]
    
    def add_messages(self, *messages: ContextMessage) -> None:
        self._messages.extend(messages)
        
    def remove_message(self, message: ContextMessage) -> None:
        self._messages.remove(message)
        
    
class ChatSession:
    """Représente une session de conversation."""
    def __init__(self,
                 cog: 'Assistant',
                 guild: discord.Guild,
                 system_prompt: str,
                 *,
                 temperature: float = 1.0,
                 max_completion_tokens: int = MAX_COMPLETION_TOKENS,
                 context_window: int = CONTEXT_WINDOW) -> None:
        self.__cog = cog
        self.guild = guild
        
        self._initial_system_prompt = system_prompt
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        self.context_window = context_window
        
        self._interactions : dict[str, ChatInteraction] = {} # Historique des messages par groupes d'interactions
        self._last_history_cleanup = datetime.now()
        
    def __repr__(self) -> str:
        return f'<ChatSession guild={self.guild.name} interactions={len(self._interactions)}>'
    
    # Propriétés
    
    @property
    def system_prompt(self) -> SystemCtxMessage:
        data = {
            'assistant_name': self.guild.me.name,
            'guild_name': self.guild.name,
            'current_time': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
            'system_prompt': self._initial_system_prompt
        }
        return SystemCtxMessage(META_SYSTEM_PROMPT(data))
    
    # Gestion des interactions

    def get_interaction(self, interaction_id: str) -> ChatInteraction | None:
        return self._interactions.get(interaction_id)
    
    def get_interactions(self, cond: Callable[[ChatInteraction], bool] = lambda _: True) -> list[ChatInteraction]:
        return [interaction for interaction in self._interactions.values() if cond(interaction)]
    
    def create_interaction(self, *messages: ContextMessage) -> ChatInteraction:
        interaction = ChatInteraction(list(messages))
        self._interactions[interaction.id] = interaction
        return interaction
    
    def remove_interaction(self, interaction_id: str) -> None:
        self._interactions.pop(interaction_id, None)
        
    def clear_interactions(self, cond: Callable[[ChatInteraction], bool] = lambda _: True) -> None:
        for interaction in self.get_interactions(cond):
            self.remove_interaction(interaction.id)
            
    def delete_old_interactions(self) -> None:
        if self._last_history_cleanup < datetime.now() - CONTEXT_CLEANUP_INTERVAL:  
            self._last_history_cleanup = datetime.now()
        else:
            return
        for interaction in self.get_interactions(lambda i: i.messages[-1].timestamp < datetime.now(pytz.utc) - CONTEXT_MAX_AGE):
            if interaction.completed:
                self.remove_interaction(interaction.id)
            
    # Contexte de conversation
    
    def get_context(self) -> Sequence[ContextMessage]:
        """Renvoie les messages du contexte de conversation."""
        inters = []
        self.delete_old_interactions()
        tokens = self.system_prompt.token_count
        interactions = self.get_interactions()[::-1]    
        for interaction in interactions:
            tokens += interaction.total_token_count
            inters.append(interaction)
            if tokens >= self.context_window:
                break
        context = [self.system_prompt] + [m for i in inters[::-1] for m in i.messages]
        return context
    
    # Complétion de l'IA
    
    async def complete(self, chat_interaction: ChatInteraction, **carryover) -> ChatInteraction:
        """Demande une complétion à l'IA."""
        messages = [m.payload for m in self.get_context()]
        
        files = carryover.get('files', [])
        tools = carryover.get('tools', [])
        
        try:
            completion = await self.__cog.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages, # type: ignore
                max_tokens=self.max_completion_tokens,
                temperature=self.temperature,
                tools=GPT_TOOLS if ENABLE_TOOLS else [], # type: ignore
                tool_choice='auto',
                timeout=30
            )
        except Exception as e:
            if 'invalid_image_url' in str(e):
                logger.exception('An error occured while completing a message.', exc_info=e)
                # On efface les interactions contenant des images et on relance la complétion
                self.clear_interactions(lambda i: i.contains_image)
                return await self.complete(chat_interaction, **carryover)
            logger.exception('An error occured while completing a message.', exc_info=e)
            raise e
        
        try:
            assistant_msg = AssistantCtxMessage.from_gpt_payload(completion)
        except ValueError as e:
            logger.exception('An error occured while parsing the completion.')
            raise e
        
        if ENABLE_TOOLS and completion.choices[0].message:
            if completion.choices[0].message.tool_calls and assistant_msg.finish_reason == 'tool_calls':
                tool_call, tool_msg = await self.call_tool(completion.choices[0].message.tool_calls[0])
                if not tool_call.function_name in tools:
                    tools.append(tool_call.function_name)
                if tool_msg:
                    files.extend(tool_msg.attachments)
                    chat_interaction.add_messages(tool_call, tool_msg)
                    return await self.complete(chat_interaction, files=files, tools=tools) # On relance la complétion avec la réponse de l'outil
                else:
                    chat_interaction.add_messages(tool_call)
        
        if not assistant_msg.content:
            if assistant_msg.finish_reason == 'content_filter':
                assistant_msg._raw_content = "**Contenu filtré par OpenAI** × Veuillez reformuler votre question."
            elif not carryover.get('retry', False):
                return await self.complete(chat_interaction, files=files, tools=tools, retry=True)

        assistant_msg.files = files
        assistant_msg.tools_used = tools
        chat_interaction.add_messages(assistant_msg)
        return chat_interaction
        
    # Outils de l'assistant
    
    async def call_tool(self, tool_call: ChatCompletionMessageToolCall) -> tuple[AssistantToolCall, ToolCtxMessage | None]:
        """Appelle un outil de l'assistant."""
        call = AssistantToolCall.from_gpt_tool_call(tool_call)
        tool_msg = None
        
        if call.function_name == 'get_user_info':
            username = call.function_arguments['user']
            key = call.function_arguments['key']
            user = self.__cog.fetch_user_from_name(self.guild, username)
            if user:
                if key:
                    info = self.__cog.get_user_info_by_key(user, key)
                    if info:
                        tool_msg = ToolCtxMessage({'user': user.name, 'key': key, 'info': info}, tool_call.id)
                    else:
                        tool_msg = ToolCtxMessage({'error': f"Information '{key}' introuvable pour l'utilisateur '{username}'."}, tool_call.id)
                else:
                    info = self.__cog.get_user_info(user)
                    tool_msg = ToolCtxMessage({'user': user.name, 'info': info}, tool_call.id)
            else:
                tool_msg = ToolCtxMessage({'error': f"Utilisateur '{username}' introuvable."}, tool_call.id)
                
        elif call.function_name == 'find_users_by_key':
            key = call.function_arguments['key']
            users = self.__cog.find_users_by_key(self.guild, key)
            if users:
                info = {user.name: self.__cog.get_user_info_by_key(user, key) for user in users}
                tool_msg = ToolCtxMessage({'key': key, 'info': info}, tool_call.id)
            else:
                tool_msg = ToolCtxMessage({'error': f"Aucun utilisateur trouvé avec l'information '{key}'."}, tool_call.id)
            
        elif call.function_name == 'set_user_info':
            username = call.function_arguments['user']
            key = call.function_arguments['key']
            value = call.function_arguments['value']
            user = self.__cog.fetch_user_from_name(self.guild, username)
            if user:
                self.__cog.set_user_info(user, key, value)
                tool_msg = ToolCtxMessage({'user': user.name, 'key': key, 'value': value}, tool_call.id)
            else:
                tool_msg = ToolCtxMessage({'error': f"Utilisateur '{username}' introuvable."}, tool_call.id)
            
        elif call.function_name == 'get_server_emojis':
            emojis = self.__cog.get_server_emojis(self.guild)
            tool_msg = ToolCtxMessage({'emojis': emojis}, tool_call.id)
        
        elif call.function_name == 'search_web_pages':
            query = call.function_arguments['query']
            lang = call.function_arguments['lang']
            num = min(call.function_arguments['num_results'], 10)
            results = self.__cog.search_web_pages(query, lang=lang, num_results=num)
            tool_msg = ToolCtxMessage({'query': query, 'results': results}, tool_call.id)
            
        if tool_msg:
            tool_msg.timestamp = call.timestamp + timedelta(seconds=1)
        return call, tool_msg
    
# COG ==========================================================================
class Assistant(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        # Table pour la configuration de l'assistant
        guild_config = dataio.DictTableBuilder(
            name='guild_config',
            default_values={
                'system_prompt': DEFAULT_SYSTEM_PROMPT,
                'temperature': 1.0
            }
        )
        self.data.link(discord.Guild, guild_config)
        
        # Table pour la mémoire de l'assistant
        assistant_memory = dataio.TableBuilder(
            '''CREATE TABLE IF NOT EXISTS assistant_memory (
                user_id INTEGER,
                key TEXT,
                value TEXT,
                PRIMARY KEY (user_id, key) ON CONFLICT REPLACE
                )'''
        )
        self.data.link('global', assistant_memory)
        
        self.openai_client = AsyncOpenAI(api_key=self.bot.config['OPENAI_API_KEY']) # type: ignore
        
        self.create_audio_transcription = app_commands.ContextMenu(
            name="Transcription audio",
            callback=self.transcript_audio_callback)
        self.bot.tree.add_command(self.create_audio_transcription)
        
        self._sessions = {}
        
    async def cog_unload(self):
        self.data.close_all()
        await self.openai_client.close()
        
    # Configurations -----------------------------------------------------------
    
    def get_guild_config(self, guild: discord.Guild) -> dict:
        return self.data.get(guild).get_dict_values('guild_config')
    
    def set_guild_config(self, guild: discord.Guild, **kwargs) -> None:
        self.data.get(guild).set_dict_values('guild_config', kwargs)
        
    # Gestions des sessions ---------------------------------------------------
    
    def get_session(self, guild: discord.Guild) -> ChatSession:
        system_prompt = self.get_guild_config(guild).get('system_prompt', DEFAULT_SYSTEM_PROMPT)
        session = self._sessions.get(guild.id)
        if not session:
            session = ChatSession(self, guild, system_prompt)
            self._sessions[guild.id] = session
        return session
    
    # Outils <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # Notes utilisateur 
    
    def fetch_user_from_name(self, guild: discord.Guild, name: str) -> discord.Member | None:
        user = discord.utils.find(lambda u: u.name == name.lower(), guild.members)
        if user:
            return user if user else None
        
        # On tente d'extraire un ID
        poss_id = re.search(r'\d{17,19}', name)
        if poss_id:
            user = discord.utils.find(lambda u: u.id == int(poss_id.group(0)), guild.members)
            return user if user else None
        
        # On cherche le membre le plus proche en nom
        members = [member.name for member in guild.members]
        closest_member = fuzzy.extract_one(name, members)
        if closest_member:
            user = discord.utils.find(lambda u: u.name == closest_member[0], guild.members)
            return user if user else None
        
        # On cherche le membre le plus proche en surnom
        nicknames = [member.nick for member in guild.members if member.nick]
        closest_nickname = fuzzy.extract_one(name, nicknames, score_cutoff=90)
        if closest_nickname:
            user = discord.utils.find(lambda u: u.nick == closest_nickname[0], guild.members)
            return user if user else None
    
    def get_user_info(self, user: discord.Member | discord.User) -> dict:
        """Retourne les informations d'un utilisateur."""
        r = self.data.get('global').fetchall('SELECT * FROM assistant_memory WHERE user_id = ?', user.id)
        return {row['key']: row['value'] for row in r}
    
    def get_user_info_by_key(self, user: discord.Member | discord.User, key: str) -> str | None:
        """Retourne une information spécifique d'un utilisateur."""
        keys = self.data.get('global').fetchall('SELECT key FROM assistant_memory WHERE user_id = ?', user.id)
        if key not in [row['key'] for row in keys]:
            closest_key = fuzzy.extract_one(key, [row['key'] for row in keys])
            if closest_key:
                key = closest_key[0]
        r = self.data.get('global').fetchone('SELECT value FROM assistant_memory WHERE user_id = ? AND key = ?', user.id, key)
        return r['value'] if r else None
    
    def find_users_by_key(self, guild: discord.Guild, key: str) -> list[discord.Member]:
        """Retourne les utilisateurs ayant une information spécifique."""
        r = self.data.get('global').fetchall('SELECT user_id FROM assistant_memory WHERE key = ?', key)
        guild_members = {member.id: member for member in guild.members}
        return [guild_members[row['user_id']] for row in r if row['user_id'] in guild_members]
    
    def set_user_info(self, user: discord.Member | discord.User, key: str, value: str | None) -> None:
        """Met à jour une clé-valeur des informations d'un utilisateur."""
        if value:
            self.data.get('global').execute('INSERT OR REPLACE INTO assistant_memory (user_id, key, value) VALUES (?, ?, ?)', user.id, key, value)
        else:
            self.data.get('global').execute('DELETE FROM assistant_memory WHERE user_id = ? AND key = ?', user.id, key) 
            
    def delete_all_user_info(self, user: discord.Member | discord.User) -> None:
        """Supprime toutes les informations d'un utilisateur."""
        self.data.get('global').execute('DELETE FROM assistant_memory WHERE user_id = ?', user.id)
        
    # Recherche web -----------------------------------------------------------
    
    def search_web_pages(self, query: str, lang: str = 'fr', num_results: int = 3) -> list[dict]:
        """Recherche des informations sur le web."""
        results = search(query, lang=lang, num_results=num_results, advanced=True) 
        return [{'url': result.url, 'title': result.title, 'description': result.description} for result in results]
        
    # Fonctions de serveur ----------------------------------------------------
    
    def get_server_emojis(self, guild: discord.Guild) -> list[str]:
        """Renvoie la liste des emojis du serveur."""
        return [f"{emoji} ({emoji.name})" for emoji in guild.emojis]
        
    # Marqueurs d'outils -------------------------------------------------------
    
    def get_tool_markers(self, used_tools: list[str]) -> str:
        """Renvoie les marqueurs d'outils utilisés."""
        markers = {
            'get_user_info': "<:search:1298816145356492842> Consultation de note",
            'find_users_by_key': "<:search_key:1298973550530793472> Recherche de notes",
            'set_user_info': "<:write:1298816135722172617> Édition de note",
            'search_web_pages': "<:sitealt:1305143458830352445> Recherche web",
        }
        return ' '.join([markers.get(tool, '') for tool in used_tools])
    
    # AUDIO -------------------------------------------------------------------
    
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
            transcript = await self.openai_client.audio.transcriptions.create(
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

        if len(transcript) > 1900:
            return await message.reply(f"**Transcription demandée par {transcript_author.mention}** :\n>>> {transcript[:1900]}...", mention_author=False)
        
        await message.reply(f"**Transcription demandée par {transcript_author.mention}** :\n>>> {transcript}", mention_author=False)
    
    async def transcript_audio_callback(self, interaction: Interaction, message: discord.Message):
        """Callback pour demander la transcription d'un message audio via le menu contextuel."""
        if interaction.channel_id != message.channel.id:
            return await interaction.response.send_message("**Action impossible** × Le message doit être dans le même salon", ephemeral=True)
        if not message.attachments:
            return await interaction.response.send_message("**Erreur** × Aucun fichier n'est attaché au message.", ephemeral=True)
        return await self.handle_audio_transcription(interaction, message, interaction.user)
    
    # Listeners ----------------------------------------------------------------
    
    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not message.guild:
            return
        if message.author.bot:
            return
        if not isinstance(message.channel, (discord.TextChannel, discord.Thread)):
            return
        if message.mention_everyone:
            return
        if not message.channel.permissions_for(message.guild.me).send_messages:
            return
        
        guild = message.guild
        session = self.get_session(guild)
        if not session:
            return
        
        if guild.me.mentioned_in(message):
            usermsg = await UserCtxMessage.from_message(message)
            interaction = session.create_interaction(usermsg)
            async with message.channel.typing():
                try:
                    await session.complete(interaction)
                except Exception as e:
                    logger.exception('An error occured while completing a message.', exc_info=e)
                    return await message.reply(f"**Erreur** × Une erreur est survenue lors de la réponse à votre message.\n-# Réessayez dans quelques instants. Si le problème persiste, faîtes `/resethistory`.", mention_author=False)
                
                completion = interaction.last_assistant_message
                if not completion:
                    return await message.reply(f"**Erreur** × Aucune réponse n'a pu être générée pour votre message.", mention_author=False)
                
                files = completion.files
                content = completion._raw_content[:1900]
                markers = self.get_tool_markers(completion.tools_used)
                if markers:
                    content += f"\n-# {markers}"
                
                await message.reply(content, mention_author=False, files=files, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False, replied_user=True))
        
        
    # COMMANDES =================================================================
    
    @app_commands.command(name='systemprompt')
    @app_commands.guild_only()
    async def system_prompt_command(self, interaction: Interaction):
        """Consulter ou modifier les instructions système de l'assistant."""
        if not isinstance(interaction.guild, discord.Guild):
            return await interaction.response.send_message("**Action impossible** × Cette commande n'est pas disponible en message privé.", ephemeral=True)
        
        session = self.get_session(interaction.guild)
        if not session:
            return await interaction.response.send_message("**Erreur** × Impossible de démarrer la session de conversation.", ephemeral=True)
        
        prompt = SystemPromptModal(session._initial_system_prompt)
        await interaction.response.send_modal(prompt)
        if await prompt.wait():
            return await interaction.response.send_message("**Action annulée** × Vous avez pris trop de temps pour répondre.", ephemeral=True)
        new_prompt = prompt.new_system_prompt.value
        if new_prompt == session.system_prompt._raw_content:
            return await interaction.response.send_message("**Information** × Les instructions système n'ont pas été modifiées.", ephemeral=True)
        
        self.set_guild_config(interaction.guild, system_prompt=new_prompt)
        
        session._initial_system_prompt = new_prompt
        session.clear_interactions()
        await interaction.followup.send(f"**Instructions système mises à jour** · Voici les nouvelles instructions :\n> *{new_prompt}*\n-# Afin d'éviter que les nouvelles instructions rentrent en conflit avec les précédents messages de l'assistant (sous d'anciennes instructions), sa mémoire a été réinitalisée.", ephemeral=True)
        
    @app_commands.command(name='temperature')
    @app_commands.guild_only()
    @app_commands.rename(temp='température')
    async def cmd_temperature(self, interaction: Interaction, temp: app_commands.Range[float, 0.0, 2.0]):
        """Modifier le degré de créativité de l'assistant.

        :param temp: Température de génération, entre 0.0 et 2.0"""
        if not isinstance(interaction.guild, discord.Guild):
            return await interaction.response.send_message("**Action impossible** × Cette commande n'est pas disponible en message privé.", ephemeral=True)
        
        session = self.get_session(interaction.guild)
        if not session:
            return await interaction.response.send_message("**Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        # On met à jour la température
        self.set_guild_config(interaction.guild, temperature=temp)
        session.temperature = temp
        if temp > 1.4:
            return await interaction.response.send_message(f"**Température mise à jour** · La température de génération est désormais de ***{temp}***.\n-# Attention, une température élevée peut entraîner des réponses incohérentes.", ephemeral=True)
        await interaction.response.send_message(f"**Température mise à jour** · La température de génération est désormais de {temp}.", ephemeral=True)
    
    @app_commands.command(name='info')
    @app_commands.guild_only()
    async def cmd_info(self, interaction: Interaction):
        """Afficher les informations sur l'assistant sur la session en cours."""
        if not isinstance(interaction.guild, discord.Guild):
            return await interaction.response.send_message("**Action impossible** × Cette commande n'est pas disponible en message privé.", ephemeral=True)
        
        embed = discord.Embed(title="Informations sur l'assistant", color=discord.Color(0x000001))
        embed.set_thumbnail(url=interaction.guild.me.display_avatar.url)
        embed.set_footer(text="Implémentation de GPT4o-mini et Whisper-1 (par OpenAI)", icon_url="https://static-00.iconduck.com/assets.00/openai-icon-2021x2048-4rpe5x7n.png")
        session = self.get_session(interaction.guild)
        if not session:
            return await interaction.response.send_message("**Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        # Informations sur l'assistant
        embed.add_field(name="Instructions", value=pretty.codeblock(session._initial_system_prompt), inline=False)
        embed.add_field(name="Température", value=pretty.codeblock(str(session.temperature), lang='css'))
        
        # Informations sur la session
        embed.add_field(name="Interactions", value=pretty.codeblock(str(len(session._interactions))))
        embed.add_field(name="Tokens mémorisés", value=pretty.codeblock(str(sum([m.token_count for m in session.get_context()]))))
        
        await interaction.response.send_message(embed=embed)
        
    @app_commands.command(name='resethistory')
    @app_commands.guild_only()
    async def cmd_resethistory(self, interaction: Interaction):
        """Réinitialiser les messages en mémoire de l'assistant."""
        if not isinstance(interaction.guild, discord.Guild):
            return await interaction.response.send_message("**Action impossible** × Cette commande n'est pas disponible en message privé.", ephemeral=True)
        
        session = self.get_session(interaction.guild)
        if not session:
            return await interaction.response.send_message("**Erreur interne** × Impossible de récupérer la session de chat.", ephemeral=True)
        
        session.clear_interactions()
        await interaction.response.send_message("**Historique réinitialisé** · L'historique des messages de la session a été effacé.", ephemeral=True)
        
    memory_group = app_commands.Group(name='memory', description="Gestion des notes internes de l'assistant")

    @memory_group.command(name='show')
    async def cmd_show_memory(self, interaction: Interaction):
        """Consulter les notes de l'assistant associées à vous."""
        user = interaction.user
        notes = self.get_user_info(user)
        if not notes:
            return await interaction.response.send_message(f"**Notes de l'assistant** · Aucune note n'est associée à vous.", ephemeral=True)
        
        text = '\n'.join(sorted([f"`{key}` → *{value}*" for key, value in notes.items()]))
        embed = discord.Embed(title=f"Notes de l'assistant", description=text, color=discord.Color(0x000001))
        embed.set_thumbnail(url=user.display_avatar.url)
        embed.set_footer(text=f"Notes gérées localement par l'assistant")
        await interaction.response.send_message(embed=embed, ephemeral=True)
        
    @memory_group.command(name='edit')
    @app_commands.rename(key='clé', value='valeur')
    async def cmd_edit_memory(self, interaction: Interaction, key: str, value: str):
        """Modifier les notes de l'assistant associées à vous.
        
        :param key: Clé de la note
        :param value: Valeur de la note"""
        user = interaction.user
        notes = self.get_user_info(user)
        self.set_user_info(user, key, value)
        if not notes:
            return await interaction.response.send_message(f"**Note ajoutée** · La note de l'assistant associée à vous pour la clé `{key}` a été ajoutée.", ephemeral=True)
        await interaction.response.send_message(f"**Note modifiée** · La note de l'assistant associée à vous pour la clé `{key}` a été mise à jour.", ephemeral=True)
        
    @memory_group.command(name='delete')
    @app_commands.rename(key='clé')
    async def cmd_delete_memory(self, interaction: Interaction, key: str | None = None):
        """Supprimer les notes de l'assistant associées à vous.
        
        :param key: Clé de la note à supprimer"""
        user = interaction.user
        if not key:
            view = ConfirmView(user)
            await interaction.response.send_message(f"**Confirmation** · Êtes-vous sûr de vouloir supprimer toutes les notes de l'assistant associées à vous ?", ephemeral=True, view=view)
            await view.wait()
            if not view.value:
                return await interaction.edit_original_response(content="**Action annulée** · Les notes de l'assistant n'ont pas été supprimées.", view=None)
            self.delete_all_user_info(user)
            return await interaction.edit_original_response(content="**Notes supprimées** · Toutes les notes de l'assistant associées à vous ont été supprimées.", view=None)
        
        notes = self.get_user_info_by_key(user, key)
        if not notes:
            return await interaction.response.send_message(f"**Notes de l'assistant** · Aucune note n'est associée à vous pour la clé `{key}`.", ephemeral=True)
        
        view = ConfirmView(user)
        await interaction.response.send_message(f"**Confirmation** · Êtes-vous sûr de vouloir supprimer la note de l'assistant associée à vous pour la clé `{key}` ?", ephemeral=True, view=view)
        await view.wait()
        if not view.value:
            return await interaction.edit_original_response(content="**Action annulée** · La note de l'assistant n'a pas été supprimée.", view=None)
        
        self.set_user_info(user, key, None)
        await interaction.edit_original_response(content=f"**Note supprimée** · La note de l'assistant associée à vous pour la clé `{key}` a été supprimée.", view=None)

    @cmd_edit_memory.autocomplete('key')
    @cmd_delete_memory.autocomplete('key')
    async def autocomplete_key_callback(self, interaction: Interaction, current: str):
        user = interaction.user
        keys = self.get_user_info(user).keys()
        fuzz = fuzzy.finder(current, keys)
        return [app_commands.Choice(name=key, value=key) for key in fuzz][:15]
    
    # COMMANDES PROPRIO ========================================================
    
    @commands.command(name='setdata')
    @commands.is_owner()
    async def cmd_setdata(self, ctx: commands.Context, user_id: int, key: str, value: str):
        """Modifie une donnée utilisateur."""
        user = self.bot.get_user(user_id)
        if not user:
            return await ctx.send(f"**Erreur** × Utilisateur introuvable.")
        self.set_user_info(user, key, value)
        await ctx.send(f"**Donnée ajoutée** · La donnée `{key} → {value}` a été ajoutée pour l'utilisateur {user}.")
        
    @commands.command(name='listdata')
    @commands.is_owner()
    async def cmd_listdata(self, ctx: commands.Context, user_id: int):
        """Liste les données d'un utilisateur."""
        user = self.bot.get_user(user_id)
        if not user:
            return await ctx.send(f"**Erreur** × Utilisateur introuvable.")
        notes = self.get_user_info(user)
        if not notes:
            return await ctx.send(f"**Notes** · Aucune note n'est associée à l'utilisateur {user}.")
        text = '\n'.join([f"`{key}` → *{value}*" for key, value in notes.items()])
        await ctx.send(f"**Notes de l'utilisateur {user}**\n{text}")
        
    @commands.command(name='deletedata')
    @commands.is_owner()
    async def cmd_deletedata(self, ctx: commands.Context, user_id: int, key: str):
        """Supprimer une donnée utilisateur."""
        user = self.bot.get_user(user_id)
        if not user:
            return await ctx.send(f"**Erreur** × Utilisateur introuvable.")
        self.set_user_info(user, key, None)
        await ctx.send(f"**Donnée supprimée** · La donnée associée à la clé `{key}` pour l'utilisateur {user} a été supprimée.")
        
async def setup(bot):
    await bot.add_cog(Assistant(bot))
