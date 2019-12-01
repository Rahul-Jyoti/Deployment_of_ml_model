import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances 
import pickle


books = pd.read_csv('data/books.csv')
ratings = pd.read_csv('data/ratings.csv')
to_read = pd.read_csv('data/to_read.csv')

# books pickle

pickle.dump(books,open('books.pkl', 'wb'))

### Removing the users who have given less than 150 ratings 

counts1 = ratings['user_id'].value_counts()
ratings = ratings[ratings['user_id'].isin(counts1[counts1 >= 150].index)]

ratings.reset_index(inplace=True)
ratings.drop('index',axis=1,inplace=True)

# ratings pickle
pickle.dump(ratings,open('ratings.pkl', 'wb'))

### Creating a pivot on ratings table with rows as user_id and columns as book_id

data_mat = ratings.pivot(index='user_id',columns='book_id',values='rating').fillna(0)

# data_mat pickle
pickle.dump(data_mat,open('data_mat.pkl', 'wb'))


from scipy.sparse import csr_matrix
data_matrix = csr_matrix(data_mat.values)

### Using KNN algorithm from scikit

model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=10, n_jobs=-1)
model_knn.fit(data_matrix)

## pickle

filename = 'model.pkl'
pickle.dump(model_knn,open(filename,'wb'))


## Take user_id as input

user_id_index = int(input("Enter user id : "))
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
            temp = ratings[ratings.user_id == k][ratings.rating == l].book_id.values.tolist()
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


for i in book_title_of_books_recommended:
    print(i)
