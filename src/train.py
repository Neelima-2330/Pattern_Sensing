from src.dataloader import get_data_generators
from src.model import build_model
import os

def main():
    data_path = 'data/raw/patterns'
    train_gen, val_gen = get_data_generators(data_path)
    num_classes = train_gen.num_classes

    model = build_model(num_classes=num_classes)

    model.fit(train_gen, validation_data=val_gen, epochs=10)

    os.makedirs('outputs', exist_ok=True)
    model.save('outputs/model.h5')
    print("✅ Model saved to outputs/model.h5")

if __name__ == '__main__':
    main()
