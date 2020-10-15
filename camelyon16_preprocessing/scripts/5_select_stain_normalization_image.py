import openslide
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--source_image_path",
                        type=str,
                        default='/data/camelyon16_original/training/normal/normal_056.tif')
    parser.add_argument("--x",
                        type=int,
                        default=120832)
    parser.add_argument("--y",
                        type=int,
                        default=23808)
    parser.add_argument("--w",
                        type=int,
                        default=2816)
    parser.add_argument("--h",
                        type=int,
                        default=2816)
    parser.add_argument("--output_stain_target_image_path",
                        type=str,
                        default='/data/camelyon16/stain_target_image.jpg')
    args = parser.parse_args()

    LEVEL = 0

    slide = openslide.OpenSlide(args.source_image_path)
    image = slide.read_region((args.y, args.x), LEVEL, (args.w, args.h)).convert('RGB')
    image.save(args.output_stain_target_image_path)
