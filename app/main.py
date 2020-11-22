from os import getenv

from cv2.cv2 import VideoCapture
from telegram import bot, Bot
from telegram.ext import Updater, run_async, MessageHandler, Filters

from app.model import StyleTransferModel

TOKEN = getenv("TOKEN")
MODELS_DIR = getenv("MODELS_DIR")
STYLE_PATH = getenv("STYLE_PATH")
WEBHOOK_URL = getenv("WEBHOOK_URL")

bot = Bot(TOKEN)
model = StyleTransferModel(MODELS_DIR)
model.load_style(STYLE_PATH)


@run_async
def document_handler(update, context):
    print("")


@run_async
def message_handler(update, context):
    bot.send_message(update.message.chat.id, "Send video file")


@run_async
def video_handler(update, context):
    video = update.message.video
    video_id = video.file_id

    video_meta = bot.get_file(video_id)

    video_meta.download("out.avi")
    video_capture = VideoCapture("out.avi")
    frames = model.inference(video_capture)
    bot.send_video(chat_id=update.message.chat.id, video=frames)


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
    updater.bot.set_webhook(url=WEBHOOK_URL)
    updater.idle()


if __name__ == '__main__':
    main()
