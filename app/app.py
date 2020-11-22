from telegram import bot, Bot
from telegram.ext import Updater, run_async, MessageHandler, Filters

TOKEN = "1449211534:AAHKKLwNpIFioaDOnolbnG8ZEpFcFUkDkbw"

bot = Bot(TOKEN)


@run_async
def document_handler(update, context):
    print("")


@run_async
def message_handler(update, context):
    print("")


@run_async
def video_handler(update, context):
    video = update.message.video
    video_id = video.file_id

    video_meta = bot.get_file(video_id)
    video_bytes = video_meta.download_as_bytearray()
    print("")


def main():
    updater = Updater(bot=bot, use_context=True)
    dispatcher = updater.dispatcher

    document = MessageHandler(Filters.document, document_handler)
    text = MessageHandler(Filters.text, message_handler)
    video = MessageHandler(Filters.video, video_handler)
    dispatcher.add_handler(document)
    dispatcher.add_handler(text)
    dispatcher.add_handler(video)

    updater.start_webhook(listen='127.0.0.1', port=8443, url_path="/")
    updater.bot.set_webhook(url="https://95e74aae94a1.ngrok.io")
    updater.idle()


if __name__ == '__main__':
    main()
