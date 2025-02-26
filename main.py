import numpy as np
import cv2


def read_image(filename: str) -> np.array:
    """
    Inputs: String relative/absolute path to image
    Outputs: 2d np array of pixel HLS color values, sorted by frequency
    """
    if filename.endswith(".png"):
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

        img = img.reshape(-1, 4)
        img = img[img[:, 3] > 250][:, :3]
        np.savetxt('my_filenmame', img, fmt='%4.6f', delimiter=' ')
        img_hls = cv2.cvtColor(np.array([img]), cv2.COLOR_RGB2HLS)
        img_hls = img_hls.reshape(-1, 3)

    else:
        img = cv2.imread(filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(-1, 3)
        img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        img_hls = img_hls.reshape(-1, 3)
    
    return img, img_hls


def get_color_frequency(img: np.array):
    """
    Inputs: Image(Nx3 array of color values)
    Outputs: Nx3 array of unique color values sorted by frequency, and corresponding frequencies
    """

    unique_elements, frequency = np.unique(img, axis=0, return_counts=True)
    sorted_indexes = np.argsort(frequency)[::-1]
    sorted_by_freq = unique_elements[sorted_indexes]

    return sorted_by_freq, np.sort(frequency)[::-1]


def main():

    img_rgb, img_hls = read_image(filename="Static/FortiClient.png")

    total_pixels = len(img_rgb)
    avg_color = np.average(img_rgb, axis=0).astype(np.uint8)
    print(f"Average Color (RGB): {avg_color}")
    print(f"Image Pixels: {total_pixels}")

    unique_image, frequencies = get_color_frequency(img=img_rgb)
    print(frequencies)

    test = np.array([[[255, 50, 50]]], dtype=np.uint8)
    test = cv2.cvtColor(test, cv2.COLOR_RGB2HLS)

    get_primary_color(img=img_hls, frequencies=frequencies, avg_color=avg_color)


def get_primary_color(img: np.array, frequencies, avg_color: np.array):
    """
    Inputs: Image (n*3 array of HLS color values)
    Image color frequency (n length np array of color frequencies)
    Image average color (1x3 NP array of RGB value)
    Outputs: Tuple containing RGB color values of primary color
    """

    # Convert to float to do math
    img = img.astype(dtype=np.float16)
    frequencies = frequencies.astype(dtype=np.float16)

    # PARAMETERS
    frequency_threshold = 0
    frequency_weight = 0.5
    saturation_threshold = 0
    saturation_weight = 0
    luminosity_min_threshold = 0
    luminosity_max_threshold = 0
    luminosity_weight = 0
    distance_from_average_threshold = 0
    distance_from_average_weight = 0
    hue_priority = "[equation to boost magenta color family and reduce tan]"
    hue_weight = 0

    frequency_weights = frequencies/len(img)*frequency_weight
    print(frequency_weights)


    """
    color_group_range = 10
    primary_min_distance_from_average = 50
    primary_max_brightness = 200
    primary_min_brightness = 70
    accent_min_distance_from_primary = 60
    accent_min_brightness = 250
    accent_min_distance_from_average = 50
    primary_freq_threshold = 0.1
    accent_freq_threshold = 0.015
    primary_stdev_threshold = 0.15
    accent_stdev_threshold = 0.07

    groupings = []

    color_channel_vals = range(round(color_group_range // 2), 255 + color_group_range // 2, color_group_range)
    flat_img = flat_img.astype(dtype=np.int16)
    for b in color_channel_vals:
        for g in color_channel_vals:
            for r in color_channel_vals:

                condition = (
                    (abs(flat_img[:, 0] - b) < color_group_range / 2) &
                    (abs(flat_img[:, 1] - g) < color_group_range / 2) &
                    (abs(flat_img[:, 2] - r) < color_group_range / 2)
                    )
                matching = flat_img[condition]
                if not matching.any():
                    continue
                group_color = np.average(matching, axis=0)

                group_color_hsv = colorsys.rgb_to_hsv(group_color[2], group_color[1], group_color[0])

                groupings.append((group_color, len(matching), (
                        group_color[0:2].std() + group_color[1:3].std() + group_color[0:3:2].std())/group_color.sum())
                                 )

                flat_img = flat_img[~condition]
                if len(flat_img) <= 1:
                    break

    primary_candidates = [
        color for color in groupings if
        color[1] > total_pixels * primary_freq_threshold and
        np.abs(color[0]).sum() < primary_max_brightness and
        color[0].std()/color[0].sum() <= primary_stdev_threshold and
        primary_min_brightness <= np.abs(color[0]).sum() and
        np.abs(color[0] - avg_color).sum() > primary_min_distance_from_average
    ]

    if not primary_candidates:
        print("check1 fail")
        primary_candidates = [
            color for color in groupings if
            primary_min_brightness <= np.abs(color[0]).sum() and
            np.abs(color[0]).sum() < primary_max_brightness and
            color[0].std()/color[0].sum() <= primary_stdev_threshold
        ]

    if not primary_candidates:
        print("check2 fail")
        primary_candidates = [
            color for color in groupings if
            primary_min_brightness <= np.abs(color[0]).sum()
        ]

    if not primary_candidates:
        primary_candidates = groupings

    primary_candidates = sorted(primary_candidates, key=lambda x: x[1], reverse=True)
    primary_color = np.round(np.copy(primary_candidates[0][0]))

    while np.abs(primary_color).sum() >= primary_max_brightness:
        primary_color[0] = primary_color[0] ** 2 / (20 + primary_color[0])
        primary_color[1] = primary_color[1] ** 2 / (20 + primary_color[1])
        primary_color[2] = primary_color[2] ** 2 / (20 + primary_color[2])

    accent_candidates = [
        color for color in groupings if
        np.abs(color[0] - primary_color).sum() > accent_min_distance_from_primary and
        np.abs(color[0] - avg_color).sum() > accent_min_distance_from_average and
        color[1] > total_pixels * accent_freq_threshold and
        # color[2] >= accent_stdev_threshold and
        np.abs(color[0]).sum() > accent_min_brightness
    ]
    if not accent_candidates:
        print("check1 fail accent")
        accent_candidates = [
            color for color in groupings if
            np.abs(color[0] - primary_color).sum() > accent_min_distance_from_primary and
            color[1] > total_pixels * accent_freq_threshold/100 and
            np.abs(color[0]).sum() > accent_min_brightness
            # color[2] >= accent_stdev_threshold
        ]

    if not accent_candidates:
        print("check2 fail accent")
        accent_candidates = [
            color for color in groupings if
            np.abs(color[0] - primary_color).sum() > accent_min_distance_from_primary and
            color[1] > total_pixels * accent_freq_threshold/100
        ]

    accent_candidates = sorted(accent_candidates, key=lambda x: x[2], reverse=True)

    accent_color = np.round(accent_candidates[0][0])

    while np.abs(accent_color).sum() <= accent_min_brightness:
        print(accent_color)
        accent_color_hsv = colorsys.rgb_to_hsv(accent_color[2], accent_color[1], accent_color[0])
        accent_color = colorsys.hsv_to_rgb(accent_color_hsv[0], accent_color_hsv[1], accent_color_hsv[2]*1.1)[::-1]

    print(avg_color)
    print(primary_color)
    print(accent_color)

    ## TESTING/REVIEWING CODE

    avg_image = np.full(
        shape=(100, 100, 3),
        fill_value=avg_color,
        dtype=np.uint8
    )
    primary_image = np.full(
        shape=(100, 100, 3),
        fill_value=primary_color,
        dtype=np.uint8
    )
    accent_image = np.full(
        shape=(100, 100, 3),
        fill_value=accent_color,
        dtype=np.uint8
    )
    cv2.imshow("average", avg_image)
    cv2.imshow("original", img)
    cv2.imshow("primary", primary_image)
    cv2.imshow("accent", accent_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """


if __name__ == "__main__":
    main()
