from My_model import Model
import get_data


def main():
    x_and_y = get_data.getXandY()
    normalization_layer = get_data.getNormalizedLayer(x_and_y[0])
    model = Model(normalization_layer)
    model.compile()
    model.fit(x_and_y[0], x_and_y[1])
    print(model.predict(6)) # 6 - это строка в /Data/New_Testing.csv

if __name__ == '__main__':
    main()