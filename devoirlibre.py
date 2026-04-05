       # 1. Trouver les variables manquantes
       # 1.1- Lisez l'ensemble de données titanic_survival.csv 
import pandas as pd
# Charger le dataset
df = pd.read_csv("titanic_survival.csv")
# Afficher les premières lignes
print(df.head())
       #1.2- la taille et informations sur  dataset
print("Taille du dataset :", df.shape)
print("Colonnes :", df.columns)
df.info()
       #1.3-Compter les  valeurs manquantes dans la colonne age 
age = df["Age"]
# Créer une série True/False pour les valeurs manquantes
age_null = age.isnull()
print(age_null)
# Sélectionner uniquement les valeurs manquantes
missing_values = age[age_null]
# Compter le nombre de valeurs manquantes
missing_valuess_count = len(missing_values)
# Afficher le résultat
print("Valeurs manquantes (age) : ",missing_valuess_count)
        #1.4-Compter les  valeurs manquantes dans la colonne cabin
print("Valeurs manquantes (cabin) :", df["Cabin"].isnull().sum())
        #1.5-Compter les valeurs manquantes pour chaque colonne
print(df.isnull().sum())


        
        #2. Gérer les variables manquantes
        # 2.1 Supprimer les lignes contenant des valeurs manquantes dans la colonne 'embarked'
df = df.dropna(subset=["Embarked"])
        # 2.2 Supprimer la colonne 'cabin'
df = df.drop(columns=["Cabin"])
        # 2.3 Imputer les valeurs manquante 
             #a) Variables numériques : remplacer les valeurs manquantes de 'age' par la moyenne
df["Age"] = df["Age"].fillna(df["Age"].mean())
             #b) Variables catégoriques : remplacer les valeurs manquantes de 'embarked' par la valeur la plus fréquente
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])



        # 3. Gérer les variables catégoriques
        # 3.1-encoder embarked avec dummies
df = pd.get_dummies(df, columns=['Embarked'])
        # 3.2-encoder sex avec map
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})



        #4. Sélection des caractéristiques
        #4.1-Sélectionner uniquement les colonnes pclass, sex, age, fare, et survived du dataset Titanic
df = df[["Pclass", "Sex", "Age", "Fare", "Survived"]]



        #5. Division de l'ensemble de donnée
        #5.1-Importer la fonction train_test_split depuis scikit-learn et diviser le dataset en un ensemble d'entrainement et un ensemble de test
from sklearn.model_selection import train_test_split
# diviser le dataset en un ensemble d'entrainement et un ensemble de test
X = df.drop("Survived", axis=1)
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)
        #5.2-la taille de chaque sous-ensemble (X_ train, X_test, y_train, y_test)
print("Taille X_train :", X_train.shape)
print("Taille X_test :", X_test.shape)
print("Taille y_train :", y_train.shape)
print("Taille y_test :", y_test.shape)



        # 6. Feature Scaling
# Importer StandardScaler, MinMaxScalerlit depuis scikit-learn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
# StandardScaler
scaler_std = StandardScaler()
X_train_std = scaler_std.fit_transform(X_train)
X_test_std = scaler_std.transform(X_test)
# MinMaxScaler
scaler_min = MinMaxScaler()
X_train_min = scaler_min.fit_transform(X_train)
X_test_min= scaler_min.transform(X_test)
# Affichage d'un exemple
print("StandardScaler - X_train :")
print(X_train_std[:5])
print("MinMaxScaler - X_train :")
print(X_train_min[:5])
