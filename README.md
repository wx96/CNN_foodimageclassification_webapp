This application is built to assist the fair pricing of food in self-service restaurant.
It was built with CNN to detect and classify rice, chicken and fish.

How it works:
1. Take a picture of the food with multiple dishes in a partitioned food container.
2. Upload the picture.
3. The classifier will detect the foood object and classify acordingly.
4. The app will then compute the total price of the meal.

How to start:
1. Install all packages required in requirements.txt
2. Run the classifier in cmd by "python classifier.py"
3. The trained model wil be saved in the root path.
4. Move the model.h5 into file models/
5. Launch the application by running "python app.py"
6. Open browser "http://localhost:5000/"

Enjoy!
