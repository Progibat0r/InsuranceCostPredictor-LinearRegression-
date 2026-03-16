# InsuranceCostPredictor
Linear Regression AI that attemps to predict insurance cost by some qualities


Описание
Это проект на Python, который позволяет предсказывать стоимость страховки через веб-интерфейс. Модель использует линейную регрессию и данные о пользователе (возраст, пол, ИМТ, курение и т.д.).

Сайт оформлен в папке InsurancePrediction/, где лежит фронтенд и серверная часть.

Ссылка на репозиторий: https://github.com/Progibat0r/InsuranceCodePredictor

-----Структура проекта-----

InsurancePrediction/ — веб-сайт проекта (HTML, CSS, PY и серверный код).

insurance.csv — датасет с характеристиками клиентов и стоимостью страховки.

lr_model.pkl — обученная модель для предсказания.

train_model.py — скрипт для обучения модели заново.

predict_model.py — скрипт для запуска модели (чтобы использовать внутри сайта).

lr_manual_dataset.py — пример работы с ручными данными.

-----Как запустить сайт-----

-Клонировать репозиторий:
git clone https://github.com/Progibat0r/InsuranceCodePredictor

-Установить зависимости 
pip install flask pandas scikit-learn matplotlib

-Перейти в папку сайта и запустить сервер:
cd InsurancePrediction
python app.py  

Открыть сайт в браузере:

http://127.0.0.1:5000
-----Как пользоваться-------

На сайте пользователь вводит данные (возраст, пол, ИМТ и другие характеристики), нажимает Submit, и получает предсказанную стоимость страховки.

🔧 Возможные улучшения
