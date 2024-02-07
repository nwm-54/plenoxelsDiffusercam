import utils

if __name__ == "__main__":
    counter = 0
    while True:
        counter += 1
        utils.capture_image()
        print(counter)