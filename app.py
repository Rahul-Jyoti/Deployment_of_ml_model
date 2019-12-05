from flask import Flask,render_template,url_for,request,jsonify
from sklearn.neighbors import NearestNeighbors
import pickle

# load the model from disk
filename = 'model.pkl'
model_knn = pickle.load(open(filename, 'rb'))
data_mat = pickle.load(open('data_mat.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
ratings = pickle.load(open('ratings.pkl', 'rb'))
app = Flask(__name__)

# @app.route("/")
# def hello():
#     return render_template('home.html')

@app.route('/',methods=['POST'])
def recommend():
    if request.method == 'POST':
        user_id_index = request.get_json(force=True)
        user_id_index = int(user_id_index)
        print("hey user_id is ",user_id_index)
        distances, indices = model_knn.kneighbors(data_mat.loc[user_id_index,:].values.reshape(1, -1), n_neighbors = 6)

        ## Finding Similar Users

        similar_users_user_id = []   ## List to store the id's of similar users

        for i in range(0, len(distances.flatten())):
            if i == 0:
                print('Recommendations for {0}:\n'.format(user_id_index))
            else:
                print('{0}: {1}, with distance of {2}:'.format(i, data_mat.index[indices.flatten()[i]], distances.flatten()[i]))
                similar_users_user_id.append(data_mat.index[indices.flatten()[i]])


        ## Rating wise recommendation dictionary

        ratings_wise_recommendation = {1:[],2:[],3:[],4:[],5:[]}

        for k in similar_users_user_id:
            for l in range(5,0,-1):
                try:
                    temp = ratings[(ratings.user_id == k) & (ratings.rating == l)].book_id.values.tolist()
                    if temp != []:
                        for i in temp:
                            ratings_wise_recommendation[l].append(i)
                except:
                    pass

        ## Book_Id's of books rated by user
        user_rated_book_id = ratings[ratings.user_id == user_id_index].book_id.values.tolist()

        ## Removing the duplicate entries of books from the rating wise recommended books dictionary
        for i in ratings_wise_recommendation:
            ratings_wise_recommendation[i] = list(set(ratings_wise_recommendation[i]))

        ## Removing the books which are already rated by user from the rating wise recommended books dictionary
        for i in ratings_wise_recommendation:
            ratings_wise_recommendation[i] = list(set(ratings_wise_recommendation[i]).difference(user_rated_book_id))

        ## Final 50 recommended book id's

        try:
            final_50_recommended_bookid_list = ratings_wise_recommendation[5][0:50]
        except:
            l = len(final_50_recommended_bookid_list)
            final_50_recommended_bookid_list.append(ratings_wise_recommendation[4][0:50-l])

        ## Title's of books recommended to user
        book_title_of_books_recommended = []
        for i in final_50_recommended_bookid_list:
            book = books[books.book_id == i].title.values.tolist()[0]
            book_title_of_books_recommended.append(book)


        # for i in book_title_of_books_recommended:
        #     print(i)    

        return jsonify(final_50_recommended_bookid_list)

if __name__ == '__main__':
    app.run(port = 5000, debug=True)
