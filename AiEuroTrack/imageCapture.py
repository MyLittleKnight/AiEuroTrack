import mss
import mss.tools


with mss.mss() as sct:
    # The screen part to capture
    monitor = {'top': 50, 'left': 64, 'width': 512, 'height': 384}
    output = 'sct-{top}x{left}_{width}x{height}.png'.format(**monitor)

    # Grab the data
    sct_img = sct.grab(monitor)

    # Save to the picture file
    mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
    print(output)
