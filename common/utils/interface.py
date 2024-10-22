# Fonctions communes pour faciliter la création d'interfaces interactives

import discord
from discord import ui, Interaction

class ConfirmationView(ui.View):
    """Ajoute deux boutons pour confirmer ou annuler une action"""
    def __init__(self, *, confirm_label: str = 'Confirmer', cancel_label: str = 'Annuler', users: list[discord.User | discord.Member], timeout: int = 30):
        super().__init__(timeout=timeout)
        self.confirm_label = confirm_label
        self.cancel_label = cancel_label
        self.users = users
        self.value = None
        
    async def interaction_check(self, interaction: Interaction):
        if interaction.user in self.users:
            return True
        await interaction.response.send_message("Vous n'avez pas la permission d'utiliser ce bouton", ephemeral=True)
        
    async def on_timeout(self) -> None:
        self.value = None
        self.stop()
        
    @ui.button(label='Confirmer', style=discord.ButtonStyle.green)
    async def confirm(self, interaction: Interaction, button: ui.Button):
        self.value = True
        self.stop()
        
    @ui.button(label='Annuler', style=discord.ButtonStyle.red)
    async def cancel(self, interaction: Interaction, button: ui.Button):
        self.value = False
        self.stop()

        
class EmbedPaginatorMenu(ui.View):
    def __init__(self, *, embeds: list[discord.Embed], 
                 users: list[discord.User | discord.Member] = [],
                 timeout: int = 60, 
                 loop: bool = False,
                 close_delete_view: bool = True):
        """Crée un menu de pagination pour des embeds

        :param embeds: Liste de discord.Embed à afficher
        :param users: Liste des utilisateurs autorisés à utiliser les boutons
        :param timeout: Temps avant que la vue ne se ferme automatiquement, par défaut 60
        :param loop: Si les pages doivent boucler, par défaut False
        :param close_delete_view: Si la vue doit être supprimée après la fermeture, par défaut True
        """
        super().__init__(timeout=timeout)
        self.embeds = embeds
        self.users = users
        self.current_page = 0
        self.loop = loop
        self.close_delete_view = close_delete_view
        
        self.initial_interaction: Interaction
        
    async def interaction_check(self, interaction: Interaction):
        if not self.users or interaction.user in self.users:
            return True
        await interaction.response.send_message("Vous n'avez pas la permission d'utiliser ce bouton", ephemeral=True)
        
    async def on_timeout(self) -> None:
        self.stop()
        if self.initial_interaction is not None:
            await self.initial_interaction.edit_original_response(view=None)
        
    async def start(self, interaction: Interaction):
        self.initial_interaction = interaction
        self.handle_buttons()
        await interaction.followup.send(embed=self.embeds[self.current_page], view=self)
        
    def handle_buttons(self):
        self.previous_button.disabled = self.current_page == 0
        self.next_button.disabled = self.current_page == len(self.embeds) - 1   
        
    @ui.button(label='←', style=discord.ButtonStyle.blurple)
    async def previous_button(self, interaction: Interaction, button: ui.Button):
        self.current_page -= 1
        if self.current_page < 0:
            if self.loop:
                self.current_page = len(self.embeds) - 1
            else:
                self.current_page = 0
        self.handle_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.current_page], view=self)
        
    @ui.button(label='Fermer', style=discord.ButtonStyle.red)
    async def stop_button(self, interaction: Interaction, button: ui.Button):
        self.stop()
        await interaction.response.edit_message(view=None)
        if self.close_delete_view:
            await interaction.delete_original_response()
    
    @ui.button(label='→', style=discord.ButtonStyle.blurple)
    async def next_button(self, interaction: Interaction, button: ui.Button):
        self.current_page += 1
        if self.current_page >= len(self.embeds):
            if self.loop:
                self.current_page = 0
            else:
                self.current_page = len(self.embeds) - 1
        self.handle_buttons()
        await interaction.response.edit_message(embed=self.embeds[self.current_page], view=self)

        
# Fonctions -------------------------------------------------------------------
    
async def ask_confirm(interaction: Interaction, 
                  content: str,
                  *,
                  embeds: list[discord.Embed] = [],
                  view: ConfirmationView | None = None,
                  default: bool = False,
                  ephemeral: bool = True) -> bool | None:
    """Envoie un message de confirmation et attend la réponse de l'utilisateur

    :param interaction: Interaction à laquelle répondre
    :param content: Contenu du message
    :param embeds: Embeds à envoyer, par défaut []
    :param view: Vue alternative à utiliser, par défaut None
    :param default: Valeur par défaut si l'utilisateur ne répond pas, par défaut False
    :param ephemeral: Si le message doit être visible uniquement par l'utilisateur, par défaut True
    :return: bool | None
    """
    if view is None:
        view = ConfirmationView(users=[interaction.user])

    msg = await interaction.followup.send(content=content, embeds=embeds, view=view, ephemeral=ephemeral, wait=True)
    r = await view.wait()
    await msg.delete()
    if r is True:
        return default
    return view.value
