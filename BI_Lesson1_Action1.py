from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")


def load_dataset():  # 加载数据集
    mnist = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=22)
    scaler = StandardScaler()  # 标准化处理
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def cart_train(x_train, x_test, y_train, y_test):  # CART模型训练和预测
    cart = DecisionTreeClassifier()
    cart.fit(x_train, y_train)
    y_pred = cart.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)
    print("CART分类结果如下：")
    print("MSE:{0}".format(mean_squared_error(y_test, y_pred)))
    print("accuracy_score:{0}".format(score))


'''
def svc_train(x_train,x_test,y_train,y_test):  # SVC模型训练和预测
    svc = SVC()
    svc.fit(x_train,y_train)
    y_pred = svc.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    score = accuracy_score(y_test, y_pred)
    print("SVC分类结果如下：")
    print("MSE:{0}".format(mean_squared_error(y_test, y_pred)))
    print("accuracy_score:{0}".format(score))
'''

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_dataset()
    cart_train(x_train, x_test, y_train, y_test)
