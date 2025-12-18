import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pywhatkit
from datetime import datetime

# ------------------ CONFIG ------------------

EMAIL_ADDRESS = "samalasaisagarreddy@gmail.com"

EMAIL_PASSWORD = "puat hcmb hhca pqvg"

TO_EMAIL = "saisagarreddy1203@gmail.com"


WHATSAPP_NUMBER = "+919676869091"  # Recipient number in international format

# -------------------------------------------

def send_email(subject, body):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print("[INFO] Email sent successfully!")
    except Exception as e:
        print(f"[ERROR] Email not sent: {e}")

import pywhatkit

def send_whatsapp_message(message):
    try:
        pywhatkit.sendwhatmsg_instantly(WHATSAPP_NUMBER, message, wait_time=10, tab_close=True)
        print("[INFO] WhatsApp message sent successfully!")
    except Exception as e:
        print(f"[ERROR] WhatsApp message not sent: {e}")

