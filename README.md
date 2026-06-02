## Встановлення залежностей

### Клонувати проект

```bash
git clone https://github.com/Walik23/e-commerce_analysis
cd e-commerce_analysis
```

### Створити віртуальне середовище

```bash
python -m venv venv
```

### Активувати віртуальне середовище

### На Windows:

```bash
venv\Scripts\activate
```

### На Linux/Mac:

```bash
source venv/bin/activate
```

### Встановити залежності

```bash
pip install -r requirements.txt
```

## Запуск системи

### Базовий запуск

```bash
python main.py
```

### Запуск тестів

```bash
pytest ./tests/ -v
```
