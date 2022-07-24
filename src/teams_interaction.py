import pyautogui
import time
import os
from datetime import datetime
import threading

# this wrapper sucks
# it will launch as a non blocking thrad the function
# but it will not allow those functions to run in parallel if
# there is other running. For that you need to add a lock object
# in the class (witht the name lock)
# and a self.lock.release() at the end of the method using this wrapper
# - the lock / unlock could be implemneted fully in the methods, but
# - it will made them to be launched all the time (i.e using resources)
# - adding directly in the decorator will not work as it will relese
# - immediately
def thread_nonparallel(func):
    def wrapper(self, *args, **kwargs):
        th = threading.Thread(target=func, args=(self, *args))
        th.start()

    return wrapper


class ButtonNotFound(Exception):
    pass


class WaitingPeriod(Exception):
    pass


class Button:
    def __init__(self, img_path, name=None, delay=0):

        if name is None:
            name = os.path.basename(img_path)

        self.name = name
        self.img_path = img_path
        self.delay = delay

        self.last_click = None

    def _process_click(self):
        button = pyautogui.locateOnScreen(self.img_path)
        if button is not None:
            pyautogui.moveTo(button)
            pyautogui.click()
        else:
            raise ButtonNotFound("Not found button on screen: " + self.name)

    def click(self):
        now = datetime.now()

        if (
            self.last_click is None
            or (now - self.last_click).total_seconds() > self.delay
        ):
            self.last_click = now
            self._process_click()
        else:
            raise WaitingPeriod("Waiting: " + self.name)


class TeamsInteractions:
    def __init__(self, use_keys_when_possible=True):

        self.lock = threading.Lock()
        self.use_keys_when_possible = use_keys_when_possible

        # CHATS

        # General buttons chats
        self.chat_open_emoji = Button("teams_img/chat_open_emoji.png")
        self.chat_send = Button("teams_img/chat_send.png")

        # Emojis within a chat session
        self.dict_emojis = {
            "smirking": Button("teams_img/chat_emo_smirking.png"),
            "grinning": Button("teams_img/chat_emo_grinning.png"),
            "expressionless": Button("teams_img/chat_emo_expressionless.png"),
        }

        # MEETINGS

        # General meeting chats
        self.meet_open_reactions = Button("teams_img/meet_open_reactions.png")
        self.meet_mute = Button("teams_img/meet_mute.png")
        self.meet_unmute = Button("teams_img/meet_unmute.png")

        # Reactions within a meeting session
        self.dict_reactions = {
            "ok": Button("teams_img/meet_reaction_ok.png"),
            "raisehand": Button("teams_img/meet_reaction_raisehand.png"),
        }

    def buttons_control(fun):
        def wrapper(self, *args, **kwargs):

            if self.lock.locked():
                return
            self.lock.acquire()

            try:
                pos = pyautogui.position()
                fun(self, *args)
                pyautogui.moveTo(pos)

            except ButtonNotFound as e:
                print(e)
            except WaitingPeriod as e:
                print(e)

            self.lock.release()

        return wrapper

    @thread_nonparallel
    @buttons_control
    def meet_call_click(self):

        if self.use_keys_when_possible:
            # It is so fast, we need to give some time so it is not pressed
            # right after
            time.sleep(2)
            pyautogui.hotkey("command", "shift", "m")
            return

        # There are two images, we need to try if it was mute, or unmute
        try:
            self.meet_mute.click()
        except ButtonNotFound:
            self.meet_unmute.click()

    @thread_nonparallel
    @buttons_control
    def chat_send_icon_click(self, icon):
        self.chat_open_emoji.click()
        time.sleep(0.2)

        self.dict_emojis[icon].click()

        time.sleep(0.05)
        self.chat_send.click()

    @thread_nonparallel
    @buttons_control
    def meet_send_reaction_click(self, reaction):

        if self.use_keys_when_possible and reaction == "raisehand":
            pyautogui.hotkey("command", "shift", "k")
            time.sleep(2)
            return

        self.meet_open_reactions.click()
        time.sleep(0.2)

        self.dict_reactions[reaction].click()

        # So that reactions windows do not stay open, lets click to close it
        time.sleep(0.1)
        self.meet_open_reactions.click()
