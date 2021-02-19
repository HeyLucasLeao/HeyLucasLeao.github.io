import smtplib
from email.message import EmailMessage
from os import environ
from time import sleep
from datetime import datetime

enderecos_de_usuarios = environ.get("ADRESS_USERS")
emails = enderecos_de_usuarios.split(',')
EMAIL_ADDRESS = environ.get("EMAIL_USER")
EMAIL_PASSWORD = environ.get("EMAIL_PASS")
data = str(datetime.now())[:10]
mensagem = ""


with open(r'estrutura.txt', "r", encoding='utf8') as f:

    for line in f:
        mensagem += line


def enviar():
    print("Enviando e-mails...")
    msg = EmailMessage()
    msg['Subject'] = f"COVID-19 BRA & AM: {data}"
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = emails
    msg.set_content(mensagem)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)
    print("E-mails enviados com sucesso.")


enviar()
sleep(10)
