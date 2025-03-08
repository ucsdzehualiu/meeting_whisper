import os
import gradio as gr
import tempfile
import shutil
import whisper
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText  # Add this import
from email.mime.base import MIMEBase
from email import encoders
from whisper_tool import get_transcription, delete_old_tmp_folders

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
model = whisper.load_model("medium", device='cuda', in_memory=True)


def send_email_with_attachment(to_email, subject, body, file_path):
    from_email = "xxxxxx@gmail"  # Replace with your email
    email_password = "xxxxxxx"  # Replace with your email password or use an app-specific password

    # Create the email message
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the body text
    msg.attach(MIMEText(body, 'plain'))  # MIMEText needs to be imported to avoid unresolved reference error

    # Attach the file
    attachment = MIMEBase('application', 'octet-stream')
    with open(file_path, 'rb') as attachment_file:
        attachment.set_payload(attachment_file.read())
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
    msg.attach(attachment)

    # Send the email
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(from_email, email_password)
        server.send_message(msg)
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")


def generate_file(file_obj, email):
    try:
        global tmpdir
        delete_old_tmp_folders()
        print('临时文件夹地址：{}'.format(tmpdir))
        print('上传文件的地址：{}'.format(file_obj.name))  # 输出上传后的文件在gradio中保存的绝对地址

        # 获取到上传后的文件的绝对路径后，其余的操作就和平常一致了

        # 将文件复制到临时目录中
        shutil.copy(file_obj.name, tmpdir)

        # 获取上传Gradio的文件名称
        FileName = os.path.basename(file_obj.name)
        print("base name", FileName)
        context = get_transcription(file_obj.name, model)
        # 获取拷贝在临时目录的新的文件地址
        print(context)
        NewfilePath = os.path.join(tmpdir, FileName)
        print(NewfilePath)

        # 在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
        outputPath = os.path.join(tmpdir, "meeting_record_" + FileName + '.txt')
        with open(outputPath, 'w') as w:
            w.write(context)

        # 通过邮箱发送结果文件
        subject = "Your Meeting Transcription"
        body = "Please find attached the transcription of your meeting."
        send_email_with_attachment(email, subject, body, outputPath)

        # 返回新文件的的地址（注意这里）
        return outputPath
    except Exception as e:
        # 获取上传Gradio的文件名称
        FileName = os.path.basename(file_obj.name)
        print("base name", FileName)
        # 获取拷贝在临时目录的新的文件地址
        NewfilePath = os.path.join(tmpdir, FileName)
        # Handle any exceptions that occur during processing
        with open(NewfilePath, 'rb') as file_obj:
            outputPath = os.path.join(tmpdir, "meeting_record_error_" + FileName + '.txt')
            with open(outputPath, 'w') as w:
                w.write(f"An error occurred: {e}")
        print(f"An error occurred: {e}")
        return None


def main():
    global tmpdir
    with tempfile.TemporaryDirectory(dir='.') as tmpdir:
        # 定义输入和输出
        inputs = [
            gr.components.File(label="Upload audio file"),
            gr.components.Textbox(label="Email", placeholder="Enter your email address")
        ]
        outputs = gr.components.File(label="Download transcription file")

        # 创建 Gradio 应用程序
        app = gr.Interface(fn=generate_file, inputs=inputs, outputs=outputs,
                           title="Meeting Transcription Generator (Whisper)",
                           description="Generates a transcription from an audio file. Once the audio file is uploaded, "
                                       "please provide your email and submit. The processed result will be sent via email.")

        # 启动应用程序
        app.launch(server_name='0.0.0.0', server_port=7859)


if __name__ == "__main__":
    main()
