import smtplib
from email.message import EmailMessage
from os import environ
from time import sleep
from datetime import datetime

from enderecos import emails
from estrutura import mensagem


EMAIL_PASSWORD = environ.get("EMAIL_PASS")
EMAIL_ADDRESS = environ.get("EMAIL_USER")

data = str(datetime.now())[:10]


def enviar():
    msg = EmailMessage()
    msg['Subject'] = f"Relatório de COVID-19 BRA & AM: {data}"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = 'ilolt.llol@gmail.com'  # emails
    msg.set_content(mensagem)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print("E-mail enviado com sucesso. \nProcesso finalizado.")
    sleep(10)


enviar()
