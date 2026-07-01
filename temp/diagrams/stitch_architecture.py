from PIL import Image, ImageDraw

r1 = Image.open("temp/diagrams/architecture_row1.png").convert("RGBA")
r2 = Image.open("temp/diagrams/architecture_row2.png").convert("RGBA")

pad = 40
arrow_h = 90
W = max(r1.width, r2.width) + pad * 2
H = r1.height + r2.height + arrow_h + pad * 3

canvas = Image.new("RGBA", (W, H), (255, 255, 255, 255))

x1 = (W - r1.width) // 2
x2 = (W - r2.width) // 2
y1 = pad
y2 = y1 + r1.height + arrow_h + pad

canvas.paste(r1, (x1, y1), r1)
canvas.paste(r2, (x2, y2), r2)

draw = ImageDraw.Draw(canvas)
cx = W // 2
ay0 = y1 + r1.height + 15
ay1 = y2 - 15
draw.line([(cx, ay0), (cx, ay1 - 25)], fill=(60, 60, 60, 255), width=6)
draw.polygon(
    [(cx - 18, ay1 - 25), (cx + 18, ay1 - 25), (cx, ay1)],
    fill=(60, 60, 60, 255),
)

canvas.convert("RGB").save("temp/diagrams/architecture_2row.png")
print("saved", canvas.size)
