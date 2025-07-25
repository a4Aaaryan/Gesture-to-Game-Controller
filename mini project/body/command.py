from datetime import datetime
import pydirectinput
from threading import Timer
import time


class CommandProcessor:
    def __init__(self):
        self.commands = []
        self.pressing_key = None
        self.pressing_timer = None

    def release_previous_key(self):
        if self.pressing_key:
            previous_key = self.pressing_key["key"]
            # print(f"releasing {previous_key}")
            pydirectinput.keyUp(previous_key)
            self.pressing_key = None

    # Clear log commands
    def limit_commands(self):
        if len(self.commands) > 900:
            self.commands = self.commands[-5:]

    def add_command(
        self,
        command,
        keyboard_enabled: bool,
        command_key_mappings: dict,
        pressing_timer_interval: float,
    ):
        self.limit_commands()

        now = datetime.now()
        self.commands.insert(0, dict(command=command, time=now))

        if keyboard_enabled:
            if command in command_key_mappings:
                key = command_key_mappings[command]
                # get current pressing key
                previous_key = None
                if self.pressing_key:
                    previous_key = self.pressing_key["key"]

                # clear old timer
                if self.pressing_timer and self.pressing_timer.is_alive():
                    # print("cancel timer")
                    self.pressing_timer.cancel()

                # new action
                if previous_key != key:
                    self.release_previous_key()
                    if key:
                        print("pressing", key)
                        pydirectinput.keyDown(key)

                if key:
                    # create new timer
                    self.pressing_timer = Timer(
                        pressing_timer_interval,
                        self.release_previous_key,
                    )
                    self.pressing_timer.start()

                    self.pressing_key = dict(key=key, time=now)
                    
                if key == "move_cursor_left":
                    t1 = time.time()
                    while(time.time() < t1 + 0.18):
                        pydirectinput.move(-200, 0)
                if key == "move_cursor_right":
                    t1 = time.time()
                    while(time.time() < t1 + 0.18):
                        pydirectinput.move(200, 0)

    def __str__(self):
        commands_list = list(map(lambda c: c["command"], self.commands))
        if not commands_list:
            return ""
        return commands_list[0] + "\n" + ", ".join(commands_list[1:20])

