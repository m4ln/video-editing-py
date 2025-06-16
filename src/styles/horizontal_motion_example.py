import random


def mosh_frames2(frames):
    for frame in frames:
        if not frame:
            continue

        for row in frame:
            for col in row:
                # only apply the effect randomly often enough
                if random.random() < 0.05:
                    if random.choice([True, False]):
                        # Randomly set horizontal component to 0 for vertical motion only
                        col[0] = 0
                    else:
                        # Randomly set vertical component to 0 for horizontal motion only
                        col[1] = 0

    return frames


def mosh_frames(frames):
    for frame in frames:
        if not frame:
            continue

        for row in frame:
            for col in row:
                # only apply the effect randomly often enough
                if random.random() < 0.1:
                    # Alternatively, you can set both components to 0 for no motion
                    col[0] = 0
                    col[1] = 0

    return frames


def mosh_frames1(frames):
    # This function randomly sets either the horizontal or vertical component of each motion vector to 0
    for frame in frames:
        if not frame:
            continue

        for row in frame:
            for col in row:
                # Randomly set horizontal or vertical component to 0
                if random.choice([True, False]):
                    col[1] = 0
                else:
                    col[0] = 0

    return frames
