{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import OneHotEncoder, StandardScaler\n",
    "from sklearn.naive_bayes import GaussianNB\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Spliting Dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = data.drop(columns=['LUNG_CANCER'])  # Replace 'target' with actual target column name\n",
    "y = data['LUNG_CANCER']  # Replace 'target' with actual target column name\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=50, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Naive Bayes "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
      "0       1   69        1               2        2              1   \n",
      "1       1   74        2               1        1              1   \n",
      "2       0   59        1               1        1              2   \n",
      "3       1   63        2               2        2              1   \n",
      "4       0   63        1               2        1              1   \n",
      "\n",
      "   CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  COUGHING  \\\n",
      "0                1         2         1         2                  2         2   \n",
      "1                2         2         2         1                  1         1   \n",
      "2                1         2         1         2                  1         2   \n",
      "3                1         1         1         1                  2         1   \n",
      "4                1         1         1         2                  1         2   \n",
      "\n",
      "   SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  LUNG_CANCER  \n",
      "0                    2                      2           2            1  \n",
      "1                    2                      2           2            1  \n",
      "2                    2                      1           2            0  \n",
      "3                    1                      2           2            0  \n",
      "4                    2                      1           1            0  \n",
      "Naive Bayes Accuracy: 94.000000%\n"
     ]
    }
   ],
   "source": [
    "# Load dataset\n",
    "data = pd.read_csv('lungcancer.csv')\n",
    "print(data.head())\n",
    "# Standardize the dataset\n",
    "scaler = StandardScaler()\n",
    "X_train = scaler.fit_transform(X_train)\n",
    "X_test = scaler.transform(X_test)\n",
    "\n",
    "# Naive Bayes\n",
    "nb = GaussianNB()\n",
    "nb.fit(X_train, y_train)\n",
    "y_pred_nb = nb.predict(X_test)\n",
    "accuracy_nb = accuracy_score(y_test, y_pred_nb)\n",
    "print(f'Naive Bayes Accuracy: {accuracy_nb * 100:f}%')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>GENDER</th>\n",
       "      <th>AGE</th>\n",
       "      <th>SMOKING</th>\n",
       "      <th>YELLOW_FINGERS</th>\n",
       "      <th>ANXIETY</th>\n",
       "      <th>PEER_PRESSURE</th>\n",
       "      <th>CHRONIC DISEASE</th>\n",
       "      <th>FATIGUE</th>\n",
       "      <th>ALLERGY</th>\n",
       "      <th>WHEEZING</th>\n",
       "      <th>ALCOHOL CONSUMING</th>\n",
       "      <th>COUGHING</th>\n",
       "      <th>SHORTNESS OF BREATH</th>\n",
       "      <th>SWALLOWING DIFFICULTY</th>\n",
       "      <th>CHEST PAIN</th>\n",
       "      <th>LUNG_CANCER</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>M</td>\n",
       "      <td>69</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>M</td>\n",
       "      <td>74</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>F</td>\n",
       "      <td>59</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>M</td>\n",
       "      <td>63</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>F</td>\n",
       "      <td>63</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>NO</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>304</th>\n",
       "      <td>F</td>\n",
       "      <td>56</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>305</th>\n",
       "      <td>M</td>\n",
       "      <td>70</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>306</th>\n",
       "      <td>M</td>\n",
       "      <td>58</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>307</th>\n",
       "      <td>M</td>\n",
       "      <td>67</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>308</th>\n",
       "      <td>M</td>\n",
       "      <td>62</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>YES</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>309 rows × 16 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "    GENDER  AGE  SMOKING  YELLOW_FINGERS  ANXIETY  PEER_PRESSURE  \\\n",
       "0        M   69        1               2        2              1   \n",
       "1        M   74        2               1        1              1   \n",
       "2        F   59        1               1        1              2   \n",
       "3        M   63        2               2        2              1   \n",
       "4        F   63        1               2        1              1   \n",
       "..     ...  ...      ...             ...      ...            ...   \n",
       "304      F   56        1               1        1              2   \n",
       "305      M   70        2               1        1              1   \n",
       "306      M   58        2               1        1              1   \n",
       "307      M   67        2               1        2              1   \n",
       "308      M   62        1               1        1              2   \n",
       "\n",
       "     CHRONIC DISEASE  FATIGUE   ALLERGY   WHEEZING  ALCOHOL CONSUMING  \\\n",
       "0                  1         2         1         2                  2   \n",
       "1                  2         2         2         1                  1   \n",
       "2                  1         2         1         2                  1   \n",
       "3                  1         1         1         1                  2   \n",
       "4                  1         1         1         2                  1   \n",
       "..               ...       ...       ...       ...                ...   \n",
       "304                2         2         1         1                  2   \n",
       "305                1         2         2         2                  2   \n",
       "306                1         1         2         2                  2   \n",
       "307                1         2         2         1                  2   \n",
       "308                1         2         2         2                  2   \n",
       "\n",
       "     COUGHING  SHORTNESS OF BREATH  SWALLOWING DIFFICULTY  CHEST PAIN  \\\n",
       "0           2                    2                      2           2   \n",
       "1           1                    2                      2           2   \n",
       "2           2                    2                      1           2   \n",
       "3           1                    1                      2           2   \n",
       "4           2                    2                      1           1   \n",
       "..        ...                  ...                    ...         ...   \n",
       "304         2                    2                      2           1   \n",
       "305         2                    2                      1           2   \n",
       "306         2                    1                      1           2   \n",
       "307         2                    2                      1           2   \n",
       "308         1                    1                      2           1   \n",
       "\n",
       "    LUNG_CANCER  \n",
       "0           YES  \n",
       "1           YES  \n",
       "2            NO  \n",
       "3            NO  \n",
       "4            NO  \n",
       "..          ...  \n",
       "304         YES  \n",
       "305         YES  \n",
       "306         YES  \n",
       "307         YES  \n",
       "308         YES  \n",
       "\n",
       "[309 rows x 16 columns]"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Decission Tree "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Decision Tree Accuracy: 98.000000%\n"
     ]
    }
   ],
   "source": [
    "dt = DecisionTreeClassifier()\n",
    "dt.fit(X_train, y_train)\n",
    "y_pred_dt = dt.predict(X_test)\n",
    "accuracy_dt = accuracy_score(y_test, y_pred_dt)\n",
    "print(f'Decision Tree Accuracy: {accuracy_dt * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Random Forest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random Forest Accuracy: 96.000000%\n"
     ]
    }
   ],
   "source": [
    "rf = RandomForestClassifier(n_estimators=100)\n",
    "rf.fit(X_train, y_train)\n",
    "y_pred_rf = rf.predict(X_test)\n",
    "accuracy_rf = accuracy_score(y_test, y_pred_rf)\n",
    "print(f'Random Forest Accuracy: {accuracy_rf * 100:f}%')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "KNN Accuracy: 92.000000%\n"
     ]
    }
   ],
   "source": [
    "knn = KNeighborsClassifier(n_neighbors=5)\n",
    "knn.fit(X_train, y_train)\n",
    "y_pred_knn = knn.predict(X_test)\n",
    "accuracy_knn = accuracy_score(y_test, y_pred_knn)\n",
    "print(f'KNN Accuracy: {accuracy_knn * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "SVM\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SVM Accuracy: 96.000000%\n"
     ]
    }
   ],
   "source": [
    "svm = SVC(kernel='linear')\n",
    "svm.fit(X_train, y_train)\n",
    "y_pred_svm = svm.predict(X_test)\n",
    "accuracy_svm = accuracy_score(y_test, y_pred_svm)\n",
    "print(f'SVM Accuracy: {accuracy_svm * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Logistic Regression \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Logistic Regression Accuracy: 96.000000%\n"
     ]
    }
   ],
   "source": [
    "lr = LogisticRegression()\n",
    "lr.fit(X_train, y_train)\n",
    "y_pred_lr = lr.predict(X_test)\n",
    "accuracy_lr = accuracy_score(y_test, y_pred_lr)\n",
    "print(f'Logistic Regression Accuracy: {accuracy_lr * 100:f}%')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Ploting Accuracy "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAA04AAAIwCAYAAACx/zuEAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjkuMiwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8hTgPZAAAACXBIWXMAAA9hAAAPYQGoP6dpAAB4T0lEQVR4nO3ddXgU19vG8XujECR4IDQluBSKBAheoLhLcfcWKEWKBLcSpEgpFHd3p1AIhQLFIcXdJQGKJBCIzvsHb/ZHSug2bchC8v1cV652z57ZfTY7bObec+aMyTAMQwAAAACAt7KxdgEAAAAA8L4jOAEAAACABQQnAAAAALCA4AQAAAAAFhCcAAAAAMACghMAAAAAWEBwAgAAAAALCE4AAAAAYAHBCQAAAAAsIDgBQBwxmUwaOnRojLe7fv26TCaT5s+fH+s1/ReLFi1Srly5ZG9vrxQpUli7HHzg3tf9HAAiEZwAJCjz58+XyWSSyWTSvn373rjfMAy5ubnJZDKpRo0aVqjw39u9e7f5tZlMJtnb2ytLlixq2bKlrl69GqvPdf78ebVu3VpZs2bVrFmzNHPmzFh9/ITK19dXzZs3l5ubmxwdHZUqVSpVqFBB8+bNU3h4uLXLA4AEzc7aBQCANSRKlEhLly5VqVKlorTv2bNHt2/flqOjo5Uq+++6deumIkWKKDQ0VMePH9fMmTO1ZcsWnTp1Sq6urrHyHLt371ZERIR++OEHZcuWLVYeM6GbPXu2vvzyS7m4uKhFixbKnj27AgMD5ePjo3bt2unevXvq37+/tct8ZzJlyqQXL17I3t7e2qUAQLQITgASpGrVqmnVqlWaPHmy7Oz+91G4dOlSeXh46OHDh1as7r8pXbq0vvjiC0lSmzZtlCNHDnXr1k0LFiyQl5fXf3rs58+fK0mSJLp//74kxeoUvaCgIDk5OcXa431IDh48qC+//FLFixfX1q1blSxZMvN93bt319GjR3X69GkrVvjuhIWFKSIiQg4ODkqUKJG1ywGAt2KqHoAEqUmTJvrzzz+1Y8cOc1tISIhWr16tpk2bRrvN8+fP1atXL/M0qpw5c+r777+XYRhR+gUHB6tHjx5KmzatkiVLplq1aun27dvRPuadO3fUtm1bubi4yNHRUZ988onmzp0bey9UUvny5SVJ165dM7f9/PPPKl26tJIkSaJkyZKpevXqOnPmTJTtWrduraRJk+rKlSuqVq2akiVLpmbNmsnd3V1DhgyRJKVNm/aNc7d++uknffLJJ3J0dJSrq6u6dOmiJ0+eRHnssmXLKm/evDp27JjKlCkjJycn9e/f33yey/fff6+pU6cqS5YscnJyUqVKlXTr1i0ZhqERI0boo48+UuLEiVW7dm09evQoymNv2LBB1atXl6urqxwdHZU1a1aNGDHijalukTWcPXtW5cqVk5OTkzJmzKixY8e+8Tt8+fKlhg4dqhw5cihRokTKkCGD6tWrpytXrpj7REREaNKkSfrkk0+UKFEiubi4qFOnTnr8+LHF92jYsGEymUxasmRJlNAUqXDhwmrdurX59j/dF00mk7p27apVq1YpT548Spw4sYoXL65Tp05JkmbMmKFs2bIpUaJEKlu2rK5fv/7W96lEiRJKnDixMmfOrOnTp0fpFxISosGDB8vDw0POzs5KkiSJSpcurV9//TVKv9ff30mTJilr1qxydHTU2bNnoz3Hyc/PT23atNFHH30kR0dHZciQQbVr136jzpjsc//k/QaA6DDiBCBBcnd3V/HixbVs2TJVrVpV0qsw8fTpUzVu3FiTJ0+O0t8wDNWqVUu//vqr2rVrpwIFCmj79u3q3bu37ty5o4kTJ5r7tm/fXosXL1bTpk1VokQJ7dq1S9WrV3+jBn9/fxUrVsx8cJs2bVr9/PPPateunQICAtS9e/dYea2RB/epU6eW9GpRh1atWqly5coaM2aMgoKCNG3aNJUqVUonTpyQu7u7eduwsDBVrlxZpUqV0vfffy8nJye1bt1aCxcu1Lp16zRt2jQlTZpUn376qSRp6NChGjZsmCpUqKCvvvpKFy5c0LRp03TkyBHt378/yjSsP//8U1WrVlXjxo3VvHlzubi4mO9bsmSJQkJC9PXXX+vRo0caO3asGjZsqPLly2v37t3q27evLl++rB9//FHffvttlLA5f/58JU2aVD179lTSpEm1a9cuDR48WAEBARo3blyU383jx49VpUoV1atXTw0bNtTq1avVt29f5cuXz7xfhIeHq0aNGvLx8VHjxo31zTffKDAwUDt27NDp06eVNWtWSVKnTp00f/58tWnTRt26ddO1a9c0ZcoUnThx4o3X/rqgoCD5+PioTJky+vjjjy2+nzHZFyVp79692rhxo7p06SJJ8vb2Vo0aNdSn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",
      "text/plain": [
       "<Figure size 1000x500 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "accuracy_scores = {\n",
    "    'Naive Bayes': accuracy_nb,\n",
    "    'Decision Tree': accuracy_dt,\n",
    "    'Random Forest': accuracy_rf,\n",
    "    'KNN': accuracy_knn,\n",
    "    'SVM': accuracy_svm,\n",
    "    'Logistic Regression': accuracy_lr\n",
    "}\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.barplot(x=list(accuracy_scores.keys()), y=list(accuracy_scores.values()))\n",
    "plt.xlabel('Model')\n",
    "plt.ylabel('Accuracy')\n",
    "plt.title('Model Performance Comparison')\n",
    "plt.xticks(rotation=45)\n",
    "plt.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
