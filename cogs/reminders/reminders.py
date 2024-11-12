import logging
from datetime import datetime
from typing import Union

import discord
from discord import Interaction, app_commands
from discord.ext import commands, tasks

from common import dataio
from common.utils import fuzzy, pretty

logger = logging.getLogger(f'MARIA.{__name__.split(".")[-1]}')

SUPPORTED_CHANNELS = Union[discord.TextChannel, discord.VoiceChannel, discord.DMChannel, discord.Thread, discord.GroupChannel]

class UserReminder:
    def __init__(self, id: int, user: discord.User | discord.Member, channel: SUPPORTED_CHANNELS, content: str, remind_at: datetime):
        self._id = id
        self.user = user
        self.channel = channel
        self.content = content
        self.remind_at = remind_at
        
    def __str__(self):
        return f'{self.user} - {self.channel} - {self.content} - {self.remind_at}'
    
    def __repr__(self):
        return f'<UserReminder user={self.user} channel={self.channel} content={self.content} remind_at={self.remind_at}>'
    
    def to_dict(self):
        return {
            'id': self._id,
            'user': self.user.id,
            'channel': self.channel.id,
            'content': self.content,
            'remind_at': self.remind_at.timestamp()
        }
    
# COG ==========================================================================
class Reminders(commands.Cog):
    """Système de rappels compatible avec l'assistant (assistant.py)"""
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.data = dataio.get_instance(self)
        
        reminders_table = dataio.TableBuilder(
            '''CREATE TABLE IF NOT EXISTS reminders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                channel_id INTEGER,
                content TEXT,
                remind_at INTEGER
            )''')
        self.data.link('global', reminders_table)
        
        self.check_reminders.start()
        
    def cog_unload(self):
        self.check_reminders.stop()
        self.data.close_all()
        
    # Loop ---------------------------------------------------------------------
    
    @tasks.loop(seconds=20)
    async def check_reminders(self):
        now = datetime.now()
        reminders = self.get_all_reminders()
        for reminder in reminders:
            if reminder.remind_at <= now:
                await self.send_reminder(reminder)
                self.remove_reminder(reminder._id)
                
    # Utils --------------------------------------------------------------------
    
    def extract_time_from_string(self, string: str) -> datetime | None:
        """Extrait une date d'une chaîne de caractères
        
        :param string: Chaîne de caractères à analyser
        :return: Date extraite ou None
        """
        now = datetime.now()
        formats = [
            '%d/%m/%Y %H:%M',
            '%d/%m/%Y %H',
            '%d/%m/%Y',
            '%d/%m %H:%M',
            '%d/%m',
            '%H:%M'
        ]
        extracted = None
        for format in formats:
            try:
                extracted = datetime.strptime(string, format)
                break
            except ValueError:
                pass
        if extracted:
            # On corrige les dates incomplètes
            if extracted.year == 1900:
                extracted = extracted.replace(year=now.year)
            if extracted.month == 1:
                extracted = extracted.replace(month=now.month)
            if extracted.day == 1:
                extracted = extracted.replace(day=now.day)
            if extracted.hour == 0:
                extracted = extracted.replace(hour=now.hour)
            if extracted.minute == 0:
                extracted = extracted.replace(minute=now.minute)
        else:
            return now
        return extracted
    
    # Gestions des rappels -----------------------------------------------------
    
    def get_all_reminders(self) -> list[UserReminder]:
        r = self.data.get('global').fetchall('SELECT * FROM reminders')
        reminders = []
        for row in r:
            user = self.bot.get_user(row['user_id'])
            if not isinstance(user, discord.User | discord.Member):
                continue
            channel = self.bot.get_channel(row['channel_id'])
            if not isinstance(channel, SUPPORTED_CHANNELS):
                continue
            remind_at = datetime.fromtimestamp(row['remind_at'])
            reminders.append(UserReminder(row['id'], user, channel, row['content'], remind_at))
        return reminders
    
    def get_reminders(self, user: discord.User | discord.Member) -> list[UserReminder]:
        r = self.data.get('global').fetchall('SELECT * FROM reminders WHERE user_id = ?', user.id)
        reminders = []
        for row in r:
            channel = self.bot.get_channel(row['channel_id'])
            if not isinstance(channel, SUPPORTED_CHANNELS):
                continue
            remind_at = datetime.fromtimestamp(row['remind_at'])
            reminders.append(UserReminder(row['id'], user, channel, row['content'], remind_at))
        return reminders
    
    def get_reminder(self, reminder_id: int) -> UserReminder | None:
        r = self.data.get('global').fetchone('SELECT * FROM reminders WHERE id = ?', reminder_id)
        if not r:
            return None
        user = self.bot.get_user(r['user_id'])
        if not isinstance(user, discord.User | discord.Member):
            return None
        channel = self.bot.get_channel(r['channel_id'])
        if not isinstance(channel, SUPPORTED_CHANNELS):
            return None
        remind_at = datetime.fromtimestamp(r['remind_at'])
        return UserReminder(r['id'], user, channel, r['content'], remind_at)
    
    def add_reminder(self, user: discord.User | discord.Member, channel: SUPPORTED_CHANNELS, content: str, remind_at: datetime):
        self.data.get('global').execute('INSERT OR REPLACE INTO reminders (user_id, channel_id, content, remind_at) VALUES (?, ?, ?, ?)', user.id, channel.id, content, remind_at.timestamp())
        
    def remove_reminder(self, reminder_id: int):
        self.data.get('global').execute('DELETE FROM reminders WHERE id = ?', reminder_id)
        
    # Envoi de rappels ---------------------------------------------------------
    
    async def send_reminder(self, reminder: UserReminder):
        text = f"*{reminder.content}*\n-# **RAPPEL** · {reminder.user.mention} · <t:{int(reminder.remind_at.timestamp())}:F>"
        await reminder.channel.send(text)
        
    # Commandes ----------------------------------------------------------------
    
    reminder_group = app_commands.Group(name='reminders', description='Gestion des rappels')
    
    @reminder_group.command(name='list')
    async def cmd_list_reminders(self, interaction: Interaction):
        """Liste vos rappels enregistrés"""
        user = interaction.user
        reminders = self.get_reminders(user)
        if not reminders:
            await interaction.response.send_message("**Aucun rappel** • Vous n'avez aucun rappel enregistré.", ephemeral=True)
            return
        # Tri par date
        reminders.sort(key=lambda r: r.remind_at)
        text = "## <:reminder:1305949302752940123> Vos rappels enregistrés :\n"
        text += '\n'.join([f"• <t:{int(reminder.remind_at.timestamp())}:d> <t:{int(reminder.remind_at.timestamp())}:t> → `{reminder.content}`" for reminder in reminders])
        await interaction.response.send_message(text, ephemeral=True)
        
    @reminder_group.command(name='add')
    @app_commands.rename(content='contenu', remind_at='date')
    async def cmd_add_reminder(self, interaction: Interaction, content: str, remind_at: str):
        """Ajouter un rappel

        :param content: Contenu du rappel
        :param remind_at: Date et heure du rappel
        """
        user = interaction.user
        channel = interaction.channel
        if not isinstance(channel, SUPPORTED_CHANNELS):
            return await interaction.response.send_message("**Erreur** × Ce type de canal n'est pas pris en charge.", ephemeral=True)
        time = self.extract_time_from_string(remind_at) 
        if not time:
            return await interaction.response.send_message("**Erreur** × Date et heure non reconnues.", ephemeral=True)
        self.add_reminder(user, channel, content, time)
        await interaction.response.send_message(f"**Rappel enregistré** • Le rappel sera envoyé <t:{int(time.timestamp())}:F> sur ce salon.", ephemeral=True)
        
    @cmd_add_reminder.autocomplete('remind_at')
    async def autocomplete_time(self, interaction: Interaction, current: str):
        date = self.extract_time_from_string(current)
        if not date:
            return []
        return [app_commands.Choice(name=date.strftime('%d/%m/%Y %H:%M'), value=date.strftime('%d/%m/%Y %H:%M'))]
    
    @reminder_group.command(name='remove')
    @app_commands.rename(reminder_id='identifiant')
    async def cmd_remove_reminder(self, interaction: Interaction, reminder_id: int):
        """Supprimer un rappel

        :param reminder_id: Identifiant du rappel
        """
        user = interaction.user
        reminder = self.get_reminder(reminder_id)
        if not reminder:
            return await interaction.response.send_message("**Erreur** × Rappel introuvable.", ephemeral=True)
        if reminder.user != user:
            return await interaction.response.send_message("**Erreur** × Vous n'êtes pas l'auteur de ce rappel.", ephemeral=True)
        self.remove_reminder(reminder_id)
        await interaction.response.send_message(f"**Rappel supprimé** • Le rappel pour le <t:{int(reminder.remind_at.timestamp())}:F> a été supprimé.", ephemeral=True)
        
    @cmd_remove_reminder.autocomplete('reminder_id')
    async def autocomplete_reminder_id(self, interaction: Interaction, current: str):
        user = interaction.user
        reminders = self.get_reminders(user)
        f = fuzzy.finder(current, reminders, key=lambda r: r.content)
        return [app_commands.Choice(name=f'{r._id} · {pretty.shorten_text(r.content, 50)}', value=r._id) for r in f][:10]
    
async def setup(bot):
    await bot.add_cog(Reminders(bot))
