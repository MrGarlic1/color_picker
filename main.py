import numpy as np
import cv2


def read_image(filename: str) -> np.array:
    """
    Inputs: String relative/absolute path to image
    Outputs: 2d np array of pixel HLS color values, sorted by frequency
    """
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    assert isinstance(img, np.ndarray), "Image not found. Check file path?"

    if max(img.shape) > 600:
        scale_factor = 600/max(img.shape)
        img = cv2.resize(
            img, (int(img.shape[0]*scale_factor), int(img.shape[1]*scale_factor)), interpolation=cv2.INTER_AREA
        )

    # Check if alpha channel is present, and if so, filter out background
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        img = img.reshape(-1, 4)
        img = img[img[:, 3] > 250][:, :3]
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.reshape(-1, 3)

    img_hls = cv2.cvtColor(np.array([img]), cv2.COLOR_RGB2HLS)
    img_hls = img_hls.reshape(-1, 3)
    
    return img, img_hls


def get_color_frequency(img: np.array, color_group_range: int):
    rgb_vals = range(round(color_group_range // 2), 255 + color_group_range // 2, color_group_range)

    groupings = np.zeros(shape=(len(rgb_vals)**3, 3), dtype=np.uint8)
    frequencies = np.zeros(shape=len(rgb_vals)**3, dtype=np.int32)
    added = 0

    img = img.astype(dtype=np.int16)
    for r in rgb_vals:
        for g in rgb_vals:
            for b in rgb_vals:

                condition = (
                    (abs(img[:, 0] - r) < color_group_range / 2) &
                    (abs(img[:, 1] - g) < color_group_range / 2) &
                    (abs(img[:, 2] - b) < color_group_range / 2)
                    )
                matching = img[condition]
                if not matching.any():
                    continue

                group_color = np.average(matching, axis=0)
                group_color = cv2.cvtColor(
                    np.array([[group_color]], dtype=np.uint8), cv2.COLOR_RGB2HLS
                ).reshape(3)

                groupings[added, :] = group_color
                frequencies[added] = len(matching)
                added += 1

                img = img[~condition]
                if len(img) <= 1:
                    break
    groupings = groupings[0:added + 1]
    frequencies = frequencies[0:added + 1]

    groupings = np.column_stack((groupings, frequencies))
    groupings = groupings[groupings[:, 3].argsort()][::-1]

    return groupings[:, 0:3], groupings[:, 3]


def get_primary_color(img: np.array, frequencies, avg_color: np.array, total_pixels: int):
    """
    Inputs: Image (n*3 array of HLS color values)
    Image color frequency (n length np array of color frequencies)
    Image average color (1x3 NP array of RGB value)
    Outputs: Tuple containing RGB color values of primary color
    """

    # Convert to float to do math
    img = img.astype(dtype=np.float32)
    frequencies = frequencies.astype(dtype=np.float32)
    frequencies = frequencies[frequencies > 1]
    img = img[0:len(frequencies), :]

    # PARAMETERS
    frequency_threshold = 0.02
    frequency_weight = 1.2
    saturation_min_threshold = 0.05*255
    saturation_max_threshold = 0.65*255
    saturation_weight = 1
    luminosity_min_threshold = 0.2*255
    luminosity_max_threshold = 0.3*255
    luminosity_weight = 0.25
    # distance_from_average_threshold = 0
    # distance_from_average_weight = 0
    # hue_priority = "[equation to boost magenta color family and reduce tan]"
    # hue_weight = 0

    # Normalized frequency score. Higher frequency results in higher score
    frequency_score = frequency_weight * (
        1 - np.power(
            np.full(shape=frequencies.shape, fill_value=1.0005),
            (total_pixels*frequency_threshold - frequencies)
        )
    )

    # Normalized saturation score. The midpoint of saturation min and max results in higher score
    saturation_score = (
            1 - 4 / (saturation_max_threshold-saturation_min_threshold)**2 *
            (img[:, 2] - (saturation_min_threshold + saturation_max_threshold)/2)**2
    )
    adjustment = 2 if max(saturation_score) < 0 else 0
    saturation_score = saturation_weight * (adjustment + saturation_score/abs(max(saturation_score)))

    luminosity_score = (
            1 - 4 / (luminosity_max_threshold-luminosity_min_threshold)**2 *
            (img[:, 1] - (luminosity_min_threshold + luminosity_max_threshold)/2)**2
    )
    adjustment = 2 if max(luminosity_score) < 0 else 0
    luminosity_score = luminosity_weight * (adjustment + luminosity_score/abs(max(luminosity_score)))

    final_score = frequency_score + saturation_score + luminosity_score

    idx = final_score.argmax()
    score = final_score[idx]
    color = img[idx, :].astype(np.uint8)

    print(img[idx, :])
    primary_color = cv2.cvtColor(np.array([[img[idx, :]]], dtype=np.uint8), cv2.COLOR_HLS2RGB)
    #print(color)
    #print(primary_color)
    print(f"Frequency Score: {frequency_score[idx]}")
    print(f"Saturation Score: {saturation_score[idx]}")
    print(f"Luminosity Score: {luminosity_score[idx]}")
    #print(score)
    #print(frequencies[idx])

    return primary_color.reshape(3)


def main():
    filename = "Static/fern.png"
    img_rgb, img_hls = read_image(filename=filename)

    total_pixels = len(img_rgb)
    avg_color = np.average(img_rgb, axis=0).astype(np.uint8)
    print(f"Average Color (RGB): {avg_color}")
    print(f"Image Pixels: {total_pixels}")

    img_hls, frequencies = get_color_frequency(img=img_rgb, color_group_range=15)

    primary_color = get_primary_color(img=img_hls, frequencies=frequencies, avg_color=avg_color, total_pixels=total_pixels)
    print(f"Primary Color (RGB): {primary_color}")

    full_image = cv2.imread(filename=filename, flags=cv2.IMREAD_UNCHANGED)

    avg_color_bgr = cv2.cvtColor(np.array([[avg_color]], dtype=np.uint8), cv2.COLOR_RGB2BGR).reshape(1, 3)
    primary_color_bgr = cv2.cvtColor(np.array([[primary_color]], dtype=np.uint8), cv2.COLOR_RGB2BGR).reshape(1, 3)
    print(primary_color)
    print(primary_color_bgr)

    cv2.imshow("Original", full_image)
    cv2.imshow("Average", np.full(shape=(100, 100, 3), fill_value=avg_color_bgr, dtype=np.uint8))
    cv2.imshow("Primary", np.full(shape=(100, 100, 3), fill_value=primary_color_bgr, dtype=np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    """
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
