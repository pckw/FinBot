import hupper
import main
#from dotenv import load_dotenv, find_dotenv

if __name__ == '__main__':
    #load_dotenv(find_dotenv())
    reloader = hupper.start_reloader('main.main')  # Replace 'your_gradio_app.main' with the function that launches your Gradio app
    main.main()