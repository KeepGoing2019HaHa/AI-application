import werobot
from werobot.replies import ImageReply
from IPython import embed
import os
from werobot.replies import ArticlesReply, Article
from multiprocessing import Process

# 输入微信公众平台请求凭证
robot = werobot.WeRoBot(token='yiwei', app_secret="2dd7cf40f5d65891bbcafdfae9887a8d")         # 写入服务器配置填写的 Token
robot.config["APP_ID"] = "wxd8a2ce0e5e0f964d"               # 写入开发者ID
robot.config["ENCODING_AES_KEY"] = "I3Ogys2ioLVxlpveLzpVw0oAl8wm0wVlmfo6TZkREme"     # 写入服务器配置填写的 EncodingAESKey
#irobot.config["App_Secret"] = "2dd7cf40f5d65891bbcafdfae9887a8d"     # 写入服务器配置填写的 EncodingAESKey
client=robot.client


from deoldify.visualize import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Your .*? set is empty.*?")
colorizer = get_image_colorizer(artistic=True)
render_factor=15
print("model start")


# 建立一个消息处理装饰器，当 handler 无论收到何种信息时都运行 hello 函数
@robot.handler
def hello(message):
    msg = message.content
    reply = "Hello! You said {}".format(msg)
    return reply

def process(source_path):
    result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=False)
    print(result_path)

@robot.image
def img(message):
    media_id = message.MediaId
    #reply = ImageReply(message=message, media_id=media_id)
    # embed()
    source_url = message.PicUrl
    source_path = "test.png"
    source_path = "test_images/{}.png".format(media_id)
    os.system("wget -O {} {}".format(source_path, source_url))
    
    #result_path = colorizer.plot_transformed_image(path=source_path, render_factor=render_factor, compare=False)
    #result_path = "test.png"
    #result_path = "result_images/test.png"
    #print("#"*100, result_path)
    
    result_url = "http://8.140.106.109/flask/?imageId={}.png".format(message.MediaId)
    background_process = Process(target=process, args=(source_path, ))
    background_process.start()
    #media_id = client.upload_media("image", open(r"{}".format(result_path), "rb"))['media_id']
    #reply = ImageReply(message=message, media_id=media_id)
    
    reply = ArticlesReply(message=message)
    article = Article(
                title="彩色化后的图片",
                description="点击进去即可看到原始图片",
                img="https://mmbiz.qpic.cn/mmbiz_png/BU6V4FVSBN53fVic0xKfhTnXmLvJDAk8TKbRfF4f8HHjRtTpZzsrGhBzKUqF8lG0XCqKZhf9WtzsSicRe2auWzNQ/0?wx_fmt=png",
                url=result_url,
    )
    reply.add_article(article)
    
    return reply

# 让服务器监听在 0.0.0.0:80
robot.config['HOST'] = '0.0.0.0'
robot.config['PORT'] = '5000'
robot.run()
