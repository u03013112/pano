# 打开 Terminal（终端）应用程序。
# 使用 cd 命令导航到包含您要拼接的图像的文件夹。例如：
# cd /path/to/your/images
# 使用 Hugin 的 pto_gen 命令创建一个新的项目文件。例如：
# /Applications/Hugin.app/Contents/MacOS/pto_gen -o project.pto image1.jpg image2.jpg
# 这将创建一个名为 "project.pto" 的项目文件，其中包含您要拼接的图像 "image1.jpg" 和 "image2.jpg"。
# 使用 cpfind 命令查找控制点。在此步骤中，您可以指定要使用的特征检测算法。例如，要使用 SURF 算法，请输入：
# /Applications/Hugin.app/Contents/MacOS/cpfind --method=surf -o project.pto project.pto
# 这将更新 "project.pto" 文件，包含使用 SURF 算法找到的控制点。
# 接下来，您可以继续使用 Hugin 的其他命令行工具进行图像拼接。例如，使用以下命令进行优化、拼接和生成最终图像：
# /Applications/Hugin.app/Contents/MacOS/autooptimiser -a -m -l -s -o project.pto project.pto
# /Applications/Hugin.app/Contents/MacOS/pto_var --set=yaw,pitch,roll --opt=auto project.pto
# /Applications/Hugin.app/Contents/MacOS/pto2mk -o project.mk -p final project.pto
# make -f project.mk all
# 这将生成名为 "final.tif" 的拼接图像。