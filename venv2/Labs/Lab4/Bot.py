from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, CallbackQueryHandler, MessageHandler, filters

BOT_TOKEN = '7787810470:AAEsKvJeCZgDTofdGw7diohFG8QMW_g5Yfg'


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("Option 1", callback_data='option1')],
        [InlineKeyboardButton("Option 2", callback_data='option2')],
        [InlineKeyboardButton("Option 3", callback_data='option3')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Choose an option or type a message:", reply_markup=reply_markup)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    await query.edit_message_text(text=f"You selected: {query.data}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.lower()
    if "hello" in text:
        await update.message.reply_text("Hello there!")
    elif "bye" in text:
        await update.message.reply_text("Goodbye!")
    elif "how are you" in text:
        await update.message.reply_text("I'm doing well, thank you!")
    else:
        await update.message.reply_text("I didn't understand that.")


async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Sorry, I didn't understand that command.")


if __name__ == '__main__':
    application = ApplicationBuilder().token(BOT_TOKEN).build()

    start_handler = CommandHandler('start', start)
    button_handler = CallbackQueryHandler(button)
    message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message) #Handles text messages that aren't commands
    unknown_handler = MessageHandler(filters.COMMAND, unknown)

    application.add_handler(start_handler)
    application.add_handler(button_handler)
    application.add_handler(message_handler)
    application.add_handler(unknown_handler)

    application.run_polling()
