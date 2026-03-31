from simulation import run_sim
import base64
from PIL import Image
from io import BytesIO
import webbrowser

rho = int(input('Enter Form Change Factor (0 - 10): ')) / 10
mu = int(input('Enter Shock Factor (0 - 10): ')) / 10

result = run_sim(rho=rho, mu=mu)

img_data = base64.b64decode(result['group_stage_image'])
img = Image.open(BytesIO(img_data))
img.show()

with open("bracket.html", "w", encoding="utf-8") as f:
    f.write(result['bracket_html'])

webbrowser.open("bracket.html")
