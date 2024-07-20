import random
import time

import pygame

__WARN_SOUND = 'assets/sounds/warn.mp3'
DISPLAY_SIZE = (600, 600)


class __Size:
    ICON_SIZE = (64, 64)
    CIRCLE_RADIUS_FRAME = 275
    CIRCLE_RADIUS_FRAME_WIDTH = 20
    CIRCLE_RADIUS_LED = 255
    CIRCLE_RADIUS_LED_WIDTH = 5
    CIRCLE_RADIUS_SCREEN = 250
    INFO_BOX_SHADOW = (242, 127)
    INFO_BOX_LED = (238, 123)
    INFO_BOX_LED_WIDTH = 4
    INFO_BOX = (240, 125)
    STEERING_WHEEL = (600, 100)


class __Color:
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    SNOW = (255, 250, 250)
    GREY = (30, 30, 30)
    GREY_2 = (136, 136, 136)
    SPEED_SCREEN = (240, 231, 211)
    GREY_SCREEN = (215, 215, 215)
    PURPLE_LED = (153, 50, 204)
    ORANGE_LED = (240, 188, 32)


class __Font:
    FONT_INFOBOX = 'assets/fonts/tech_font.ttf'
    FONT_DATETIME = 'assets/fonts/digital_7.ttf'
    FONT_SPEED = 'assets/fonts/technology_bold.ttf'
    FONT_KMH = 'assets/fonts/archivo_black_regular.ttf'
    FONT_INFOBOX_SIZE = 13
    FONT_DATETIME_SIZE = 25
    FONT_SPEED_SIZE = 175
    FONT_KMH_SIZE = 25


class __Position:
    TRAPEZOID_VERTICES = [(100, 450), (575, 300), (500, 600), (50, 600)]
    CIRCLE_CENTER = (DISPLAY_SIZE[0] // 2, DISPLAY_SIZE[1] // 2)
    INFO_BOX_SHADOW = (179, 384)
    INFO_BOX = (180, 385)
    INFO_BOX_LED = (181, 386)
    INFO_BOX_TEXT_1 = (DISPLAY_SIZE[0] // 2, 432)
    INFO_BOX_TEXT_2 = (DISPLAY_SIZE[0] // 2, 462)
    TIME = (25, 25)
    DATE = (DISPLAY_SIZE[0] - 25, 25)
    SPEED_TEXT = (DISPLAY_SIZE[0] // 2, 250)
    KMH_TEXT = (450, 275)
    ICON_0 = (110, 260)
    ICON_1 = (110, 225)
    ICON_2 = (110, 295)
    STEERING_WHEEL = (0, 500)


class __InfoMessage:
    NO_MESSAGE = ' '
    DEFAULT_MESSAGE = 'no message available'
    INIT_MESSAGE = 'ISA-System initialising...'
    STANDBY_MESSAGE_1 = 'ISA-System in Standby'
    STANDBY_MESSAGE_2 = 'active from 30 to 180 km/h'
    ACTIVE_MESSAGE_1 = 'ISA-System active'
    ACTIVE_MESSAGE_WARN = 'SPEED limit exceeded!'
    INACTIVE_MESSAGE_1 = 'ISA system disabled'
    INACTIVE_MESSAGE_2 = 'Press button to activate'
    INACTIVE_MESSAGE_ERROR = 'ERROR: Camera inactive!'


class __Icon:
    CAMERA_OFF = 'assets/icons/camera_off.png'
    SYSTEM_ACTIVE = 'assets/icons/system_active.png'
    SPEED_EXCEEDED = 'assets/icons/speed_exceed.png'
    END_LIMIT = 'assets/icons/end_limit.png'
    THIRTY = 'assets/icons/30.png'
    FIFTY = 'assets/icons/50.png'
    SIXTY = 'assets/icons/60.png'
    SEVENTY = 'assets/icons/70.png'
    EIGHTY = 'assets/icons/80.png'
    HUNDRED = 'assets/icons/100.png'
    HUNDRED_TWENTY = 'assets/icons/120.png'
    STEERING_WHEEL = 'assets/icons/steering_wheel.png'


def __draw_text_infobox(screen, info_text_1, info_text_2=__InfoMessage.NO_MESSAGE):
    """
    Draws text on the dashboard's info box.

    Parameters:
    screen (pygame.Surface): The surface on which the text will be drawn.
    info_text_1 (str): The primary text to be displayed in the info box.
    info_text_2 (str): The secondary text to be displayed in the info box. Defaults to __InfoMessage.NO_MESSAGE.

    Returns:
    None. The function draws text on the provided surface.
    """
    font = pygame.font.Font(__Font.FONT_INFOBOX, __Font.FONT_INFOBOX_SIZE)
    info_text = font.render(info_text_1, True, __Color.GREY)
    info_text_rect = info_text.get_rect()
    (info_text_rect.centerx, info_text_rect.centery) = __Position.INFO_BOX_TEXT_1
    error_text = font.render(info_text_2, True, __Color.GREY)
    error_text_rect = error_text.get_rect()
    (error_text_rect.centerx, error_text_rect.centery) = __Position.INFO_BOX_TEXT_2
    screen.blit(error_text, error_text_rect)
    screen.blit(info_text, info_text_rect)


def __draw_icon(screen, icon, size, position):
    """
    Draws an icon on the provided surface at the specified position and scales it to the given size.

    Parameters:
    screen (pygame.Surface): The surface on which the icon will be drawn.
    icon (str): The path to the image file of the icon.
    size (tuple): A tuple representing the (width, height) of the icon after scaling.
    position (tuple): A tuple representing the (x, y) coordinates of the top-left corner of the icon on the surface.

    Returns:
    None. The function draws the icon on the provided surface.
    """
    icon = pygame.transform.scale(pygame.image.load(icon).convert_alpha(), size)
    screen.blit(icon, position)


def __get_speed_icon(speed_limit):
    if speed_limit == 30:
        return __Icon.THIRTY
    elif speed_limit == 50:
        return __Icon.FIFTY
    elif speed_limit == 60:
        return __Icon.SIXTY
    elif speed_limit == 70:
        return __Icon.SEVENTY
    elif speed_limit == 80:
        return __Icon.EIGHTY
    elif speed_limit == 100:
        return __Icon.HUNDRED
    elif speed_limit == 120:
        return __Icon.HUNDRED_TWENTY
    elif speed_limit == 999:
        return __Icon.END_LIMIT


def __draw_dashboard(screen, current_speed, current_date, current_time):
    """
    Draws a dashboard with a tachometer, date, time, and a speed limit indicator.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    current_date (str): The current date in the format "dd.mm.yyyy".
    current_time (str): The current time in the format "HH:MM".

    Returns:
    None. The function draws the dashboard on the provided surface.
    """
    # Fill the background with black
    screen.fill(__Color.BLACK)

    # Shadow
    pygame.draw.polygon(screen, __Color.GREY, __Position.TRAPEZOID_VERTICES)

    # Tachometer-Frame    
    pygame.draw.circle(screen, __Color.GREY, __Position.CIRCLE_CENTER, __Size.CIRCLE_RADIUS_FRAME,
                       __Size.CIRCLE_RADIUS_FRAME_WIDTH)

    # Tachometer-LED 
    pygame.draw.circle(screen, __Color.PURPLE_LED, __Position.CIRCLE_CENTER, __Size.CIRCLE_RADIUS_LED,
                       __Size.CIRCLE_RADIUS_LED_WIDTH)

    # Tachometer-Screen    
    pygame.draw.circle(screen, __Color.SPEED_SCREEN, __Position.CIRCLE_CENTER, __Size.CIRCLE_RADIUS_SCREEN)

    # Information Box with shadow and LED
    rect_outer_shadow = pygame.Rect(__Position.INFO_BOX_SHADOW[0], __Position.INFO_BOX_SHADOW[1],
                                    __Size.INFO_BOX_SHADOW[0], __Size.INFO_BOX_SHADOW[1])
    rect_outer_rect = pygame.Rect(__Position.INFO_BOX[0], __Position.INFO_BOX[1], __Size.INFO_BOX[0],
                                  __Size.INFO_BOX[1])
    rect_inner_rect = pygame.Rect(__Position.INFO_BOX_LED[0], __Position.INFO_BOX_LED[1], __Size.INFO_BOX_LED[0],
                                  __Size.INFO_BOX_LED[1])
    pygame.draw.rect(screen, __Color.GREY_2, rect_outer_shadow)
    pygame.draw.rect(screen, __Color.GREY_SCREEN, rect_outer_rect)
    pygame.draw.rect(screen, __Color.ORANGE_LED, rect_inner_rect, __Size.INFO_BOX_LED_WIDTH)

    # Time
    font_dt = pygame.font.Font(__Font.FONT_DATETIME, __Font.FONT_DATETIME_SIZE)
    time_text = font_dt.render(current_time, True, __Color.SNOW)
    screen.blit(time_text, __Position.TIME)

    # Date
    date_text = font_dt.render(current_date, True, __Color.SNOW)
    date_text_rect = date_text.get_rect()
    (date_text_rect.right, date_text_rect.top) = __Position.DATE
    screen.blit(date_text, date_text_rect)

    # Speed
    font_speed = pygame.font.Font(__Font.FONT_SPEED, __Font.FONT_SPEED_SIZE)
    font_kmh = pygame.font.Font(__Font.FONT_KMH, __Font.FONT_KMH_SIZE)
    speed_text = font_speed.render(f"{current_speed}", True, __Color.BLACK)
    speed_text_rect = speed_text.get_rect()
    (speed_text_rect.centerx, speed_text_rect.centery) = __Position.SPEED_TEXT
    kmh_text = font_kmh.render(f"km/h", True, __Color.BLACK)
    kmh_text_rect = kmh_text.get_rect()
    (kmh_text_rect.centerx, kmh_text_rect.centery) = __Position.KMH_TEXT
    screen.blit(speed_text, speed_text_rect)
    screen.blit(kmh_text, kmh_text_rect)


def display_dash_init(screen, current_speed=0, current_date='', current_time=''):
    """
    This function displays the initial dashboard state.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int, optional): The current speed of the vehicle. Defaults to 0.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the initial dashboard state on the provided surface.
    """
    __draw_dashboard(screen, current_speed, current_date, current_time)
    __draw_text_infobox(screen, __InfoMessage.INIT_MESSAGE)


def display_dash_standby(screen, current_speed, current_date='', current_time=''):
    """
    This function displays the standby dashboard state.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the standby dashboard state on the provided surface.
    """
    __draw_dashboard(screen, current_speed, current_date, current_time)
    __draw_text_infobox(screen, __InfoMessage.STANDBY_MESSAGE_1, __InfoMessage.STANDBY_MESSAGE_2)


def display_dash_active(screen, current_speed, speed_limit=-1, current_date='', current_time=''):
    """
    This function displays the active dashboard state. It also draws the end of limit sign.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    speed_limit (int, optional): The speed limit for the current location. Defaults to -1.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the active dashboard state on the provided surface.
    """
    __draw_dashboard(screen, current_speed, current_date, current_time)
    __draw_text_infobox(screen, __InfoMessage.ACTIVE_MESSAGE_1)
    __draw_icon(screen, __Icon.SYSTEM_ACTIVE, __Size.ICON_SIZE, __Position.ICON_1)
    if speed_limit >= 30:
        icon = __get_speed_icon(speed_limit)
        __draw_icon(screen, icon, __Size.ICON_SIZE, __Position.ICON_2)


def display_dash_inactive(screen, current_speed, current_date='', current_time=''):
    """
    This function displays the inactive dashboard state.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the inactive dashboard state on the provided surface.
    """
    __draw_dashboard(screen, current_speed, current_date, current_time)
    __draw_text_infobox(screen, __InfoMessage.INACTIVE_MESSAGE_1, __InfoMessage.INACTIVE_MESSAGE_2)
    __draw_icon(screen, __Icon.CAMERA_OFF, __Size.ICON_SIZE, __Position.ICON_0)


def display_dash_error(screen, current_date='', current_time=''):
    """
    This function displays the inactive dashboard state.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the inactive dashboard state on the provided surface.
    """
    __draw_dashboard(screen, 0, current_date, current_time)
    __draw_text_infobox(screen, __InfoMessage.INACTIVE_MESSAGE_1, __InfoMessage.INACTIVE_MESSAGE_ERROR)
    __draw_icon(screen, __Icon.CAMERA_OFF, __Size.ICON_SIZE, __Position.ICON_0)


def display_dash_warning_visual(screen, current_speed, speed_limit, current_date='', current_time=''):
    """
    This function displays the warning dashboard state with visual feedback.
    In warning conditions, the end of limit sign isn't displayed.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    speed_limit (int): The speed limit for the current location.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the warning dashboard state with visual feedback on the provided surface.
    """
    __draw_dashboard(screen, current_speed, current_date, current_time)
    __draw_text_infobox(screen, __InfoMessage.ACTIVE_MESSAGE_1, __InfoMessage.ACTIVE_MESSAGE_WARN)
    __draw_icon(screen, __Icon.SPEED_EXCEEDED, __Size.ICON_SIZE, __Position.ICON_1)
    if 30 <= speed_limit < 999:
        icon = __get_speed_icon(speed_limit)
        __draw_icon(screen, icon, __Size.ICON_SIZE, __Position.ICON_2)


def display_dash_warning_auditive(screen, current_speed, speed_limit, current_date='', current_time=''):
    """
    This function displays the warning dashboard state with auditive feedback.
    It first calls the `display_dash_warning_visual` function to draw the visual part of the warning state.
    Then, it initializes the pygame mixer and plays a warning sound.
    In warning conditions, the end of limit sign isn't displayed.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    speed_limit (int): The speed limit for the current location.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the warning dashboard state with auditive feedback on the provided surface.
    """
    display_dash_warning_visual(screen, current_speed, speed_limit, current_date, current_time)
    pygame.mixer.init()
    sound = pygame.mixer.Sound(__WARN_SOUND)
    sound.play()


def display_dash_warning_haptic(screen, current_speed, speed_limit, current_date='', current_time=''):
    """
    This function displays the warning dashboard state with haptic feedback.
    It first calls the `display_dash_warning_auditive` function to draw the visual part of the warning state.
    Then, it initializes the pygame mixer and plays a warning sound.
    Additionally, it applies a small random horizontal offset to the steering wheel icon to simulate haptic feedback.
    In warning conditions, the end of limit sign isn't displayed.

    Parameters:
    screen (pygame.Surface): The surface on which the dashboard will be drawn.
    current_speed (int): The current speed of the vehicle.
    speed_limit (int): The speed limit for the current location.
    current_date (str, optional): The current date in the format "dd.mm.yyyy". Defaults to an empty string.
    current_time (str, optional): The current time in the format "HH:MM". Defaults to an empty string.

    Returns:
    None. The function draws the warning dashboard state with haptic feedback on the provided surface.
    """
    start_time = time.time()
    while time.time() - start_time < 0.01:
        offset_x = random.randint(-4, 4)
        display_dash_warning_auditive(screen, current_speed, speed_limit, current_date, current_time)
        steering_wheel = pygame.transform.scale(pygame.image.load(__Icon.STEERING_WHEEL).convert_alpha(),
                                                __Size.STEERING_WHEEL)
        screen.blit(steering_wheel, (__Position.STEERING_WHEEL[0] + offset_x, __Position.STEERING_WHEEL[1]))
