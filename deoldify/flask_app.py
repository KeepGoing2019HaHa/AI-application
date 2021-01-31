from flask import Flask, send_file
from deoldify.visualize import *
import warnings


app = Flask(__name__)

# plt.style.use('dark_background')
# torch.backends.cudnn.benchmark=True
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")


colorizer = get_image_colorizer(artistic=True)
render_factor=15

@app.route('/')
def index():
	# return "hello word"
    print('start work')
    source_path = 'test_images/image.png'
    #result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=True)
    #print(result_path)
    #return send_file(result_path, mimetype='image/png')
    return "hello"

# @app.route('/get')
# def run():
#     source_path = 'test_images/image.png'
#     result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=True)
#     return send_file(result_path, mimetype='image/png')


if __name__ == "__main__":
	app.run()
