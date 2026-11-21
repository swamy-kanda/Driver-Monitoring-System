from telegram_alerts import send_telegram_message

send_telegram_message(
    "🚨 TEST ALERT\n\n"
    "Driver Monitoring System is connected successfully!\n"
    "Telegram notification system is working.",
    event_name="test"
)