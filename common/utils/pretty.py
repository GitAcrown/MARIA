# Fonctions transverses d'aide à l'affichage

from datetime import datetime, timedelta

DEFAULT_EMBED_COLOR : int = 0x2b2d31 # Couleur pour que la barre de couleur ne soit pas visible

# Disponibles sur https://discord.gg/DGaM4yH
# A modifier si vous remplacez les emojis par défaut
DEFAULT_ICONS_EMOJIS = {
    'back': '<:iconBack:1149814142354599967>',
    'next': '<:iconNext:1149814237150052513>',
    'close': '<:iconClose:1149813971239579770>',
    'ring': '<:iconRing:1190679548287787088>'
}

# Raccourcis de formattage Discord ---------------------------

def codeblock(text: str, lang: str = '') -> str:
    """Retourne le texte sous forme d'un bloc de code

    :param text: Texte à formatter
    :param lang: Langage à utiliser, par défaut "" (aucun)
    :return: str
    """
    return f"```{lang}\n{text}\n```"

# Outils d'affichage -----------------------------------------

def bargraph(value: int | float, total: int | float, *, lenght: int = 10, use_half_bar: bool = True, display_percent: bool = False) -> str:
    """Retourne un diagramme en barres

    :param value: Valeur à représenter
    :param total: Valeur maximale possible
    :param lenght: Longueur du diagramme, par défaut 10 caractères
    :param use_half_bar: S'il faut utiliser des demi-barres pour les valeurs intermédiaires, par défaut True
    :param display_percent: S'il faut afficher le pourcentage en fin de barre, par défaut False
    :return: str
    """
    if total == 0:
        return ' '
    percent = (value / total) * 100
    nb_bars = percent / (100 / lenght)
    bars = '█' * int(nb_bars)
    if (nb_bars % 1) >= 0.5 and use_half_bar:
        bars += '▌'
    if display_percent:
        bars += f' {round(percent)}%'
    return bars

def shorten_text(text: str, max_length: int, *, end: str = '...') -> str:
    """Retourne le texte raccourci (si nécessaire) à la taille maximale indiquée

    :param text: Texte à raccourcir
    :param max_length: Longueur maximale du texte, par défaut 100 caractères
    :param end: Fin du texte, par défaut '...'
    :return: str
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(end)] + end

# Outils de manipulation du temps --------------------------------

def humanize_relative_time(time: int | float | datetime, *, from_time: int | float | datetime | None = None) -> str:
    """Retourne un string représentant le temps relatif depuis le temps indiqué (ex. Il y a 2 jours)

    :param time: Temps à transformer
    :param from_time: Temps de référence, par défaut None (heure actuelle)
    :return: str
    """
    if from_time is None:
        from_time = datetime.now()
    if isinstance(time, (int, float)):
        time = datetime.now().fromtimestamp(time)
    if isinstance(from_time, (int, float)):
        from_time = datetime.now().fromtimestamp(from_time)
        
    delta = from_time - time
    if delta.days > 0:
        return f"{delta.days} jour{'s' if delta.days > 1 else ''}"
    if delta.seconds < 60:
        return f"{delta.seconds} seconde{'s' if delta.seconds > 1 else ''}"
    if delta.seconds < 3600:
        return f"{delta.seconds // 60} minute{'s' if delta.seconds // 60 > 1 else ''}"
    return f"{delta.seconds // 3600} heure{'s' if delta.seconds // 3600 > 1 else ''}"

def humanize_absolute_time(time: int | float | datetime, *, assume_today: bool = False) -> str:
    """Retourne un string représentant le temps absolu indiqué (ex. Aujourd'hui à 15h30)

    :param time: Temps à transformer
    :param assume_today: Si True, n'affiche pas la date si elle est aujourd'hui, par défaut False
    :return: str
    """
    if isinstance(time, (int, float)):
        time = datetime.now().fromtimestamp(time)
        
    french = {
        'January': 'Janvier',
        'February': 'Février',
        'March': 'Mars',
        'April': 'Avril',
        'May': 'Mai',
        'June': 'Juin',
        'July': 'Juillet',
        'August': 'Août',
        'September': 'Septembre',
        'October': 'Octobre',
        'November': 'Novembre',
        'December': 'Décembre'
    }
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    if time.day == today.day and time.month == today.month and time.year == today.year:
        return f"Aujourd'hui à {time.strftime('%Hh%M')}" if not assume_today else f"{time.strftime('%Hh%M')}"
    elif time.day == yesterday.day and time.month == yesterday.month and time.year == yesterday.year:
        return f"Hier à {time.strftime('%Hh%M')}"
    elif time.day == tomorrow.day and time.month == tomorrow.month and time.year == tomorrow.year:
        return f"Demain à {time.strftime('%Hh%M')}"
    elif time.year != today.year:
        return f"{time.day} {french[time.strftime('%B')].lower()} {time.year} à {time.strftime('%Hh%M')}"
    else:
        return f"{time.day} {french[time.strftime('%B')].lower()} à {time.strftime('%Hh%M')}"

# Nombres et chaînes de caractères -----------------------------

def bytes_to_human_readable(size: int) -> str:
    """Retourne une chaîne de caractères représentant la taille en octets donnée

    :param size: Taille en octets
    :return: str
    """
    if size < 1024:
        return f"{size} o"
    elif size < 1024 ** 2:
        return f"{round(size / 1024, 2)} Ko"
    elif size < 1024 ** 3:
        return f"{round(size / 1024 ** 2, 2)} Mo"
    elif size < 1024 ** 4:
        return f"{round(size / 1024 ** 3, 2)} Go"
    elif size < 1024 ** 5:
        return f"{round(size / 1024 ** 4, 2)} To"
    else:
        return f"{round(size / 1024 ** 5, 2)} Po"