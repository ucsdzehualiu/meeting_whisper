import os

import gradio as gr
import tempfile
import shutil

import whisper
from whisper_tool import get_transcription, delete_old_tmp_folders
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = whisper.load_model("medium", device='cuda', in_memory=True)


def generate_file(file_obj):
    try:
        global tmpdir
        delete_old_tmp_folders()
        print('临时文件夹地址：{}'.format(tmpdir))
        print('上传文件的地址：{}'.format(file_obj.name)) # 输出上传后的文件在gradio中保存的绝对地址

        #获取到上传后的文件的绝对路径后，其余的操作就和平常一致了

        # 将文件复制到临时目录中
        shutil.copy(file_obj.name, tmpdir)

        # 获取上传Gradio的文件名称
        FileName=os.path.basename(file_obj.name)
        print("base name",FileName)
        context = get_transcription(file_obj.name,model)
        # 获取拷贝在临时目录的新的文件地址
        print(context)
        NewfilePath = os.path.join(tmpdir,FileName)
        print(NewfilePath)

        # 打开复制到新路径后的文件
        # with open(NewfilePath, 'rb') as file_obj:

        #在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
        outputPath = os.path.join(tmpdir,"meeting_record_"+FileName+'.txt')
        with open(outputPath,'w') as w:
            w.write(context)

        # 返回新文件的的地址（注意这里）
        return outputPath
    except Exception as e:
        # 获取上传Gradio的文件名称
        FileName=os.path.basename(file_obj.name)
        print("base name",FileName)
        # 获取拷贝在临时目录的新的文件地址
        NewfilePath = os.path.join(tmpdir,FileName)
        # Handle any exceptions that occur during processing
        with open(NewfilePath, 'rb') as file_obj:

            #在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
            outputPath=os.path.join(tmpdir,"meeting_record_error_"+FileName+'.txt')
            with open(outputPath,'w') as w:
                w.write(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        return None

def main():
    global tmpdir
    with tempfile.TemporaryDirectory(dir='.') as tmpdir:
        # 定义输入和输出
        inputs = gr.components.File(label="Run")
        outputs = gr.components.File(label="下载文件")

        # 创建 Gradio 应用程序g
        app = gr.Interface(fn=generate_file, inputs=inputs, outputs=outputs,   title="会议纪要生成器 meeting whisper",
                      description="生成英文纪要，等音频文件上传完成再submit，模型加载、处理时间较长，请耐心等待，不要断网，不要刷新界面 不要切屏待机 处理时间参考:45分钟录音->处理时间5分钟"
      )

        # 启动应用程序
        app.launch(server_name='0.0.0.0', server_port=7859)

if __name__=="__main__":
    main()
