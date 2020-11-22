from telegram.ext import Updater, CommandHandler, run_async

TOKEN = "1449211534:AAHKKLwNpIFioaDOnolbnG8ZEpFcFUkDkbw"


@run_async
def transfer_video(update, context):
    print("hello")


def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('transfer_video', transfer_video))
    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
