import requests
import time

BOT_TOKEN = "8826103607:AAGKIBUeaOjivnUAaqj2CSczTKBYlbuwLH8"
CHAT_ID = "1178667077"

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

last_alert_time = {}
ALERT_COOLDOWN = 30


def send_telegram_message(message, event_name="general"):

    current_time = time.time()

    if event_name in last_alert_time:
        if current_time - last_alert_time[event_name] < ALERT_COOLDOWN:
            return

    try:
        response = requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": CHAT_ID,
                "text": message
            },
            timeout=10
        )

        if response.ok:
            print("[TELEGRAM] Message sent successfully!")
            last_alert_time[event_name] = current_time
        else:
            print("[TELEGRAM ERROR]", response.text)

    except Exception as e:
        print("[TELEGRAM ERROR]", e)


def send_telegram_photo(image_path, caption="Driver Alert"):

    try:
        with open(image_path, "rb") as photo:

            response = requests.post(
                f"{TELEGRAM_API}/sendPhoto",
                data={
                    "chat_id": CHAT_ID,
                    "caption": caption
                },
                files={
                    "photo": photo
                },
                timeout=20
            )

        if response.ok:
            print("[TELEGRAM] Photo sent successfully!")
        else:
            print("[TELEGRAM ERROR]", response.text)

    except Exception as e:
        print("[TELEGRAM ERROR]", e)