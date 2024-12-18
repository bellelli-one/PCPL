
print(*sorted(list(input().split()), key=lambda x: int(x[0:])))

# bot = telebot.TeleBot("7006269357:AAH3lU40qmEfhngaQ8zUXG992obKjokWkoM")

# @bot.message_handler(commands=['start'])
# def send_welcome(message):
#     bot.reply_to(message, "Привет! Я ваш бот. Как я могу вам помочь?")

# @bot.message_handler(func=lambda message: True) 
# def echo_all(message):
#     bot.reply_to(message, message.text)

# bot.polling()

