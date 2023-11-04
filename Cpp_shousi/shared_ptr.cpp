#include <iostream>

template <typename T>
class MySharedPtr {
public:
    MySharedPtr(T* ptr) : data(ptr), refCount(new int(1)) {}

    // 复制构造函数
    MySharedPtr(const MySharedPtr<T>& other) : data(other.data), refCount(other.refCount) {
        (*refCount)++;
    }

    // 析构函数
    ~MySharedPtr() {
        (*refCount)--;
        if (*refCount == 0) {
            delete data;
            delete refCount;
        }
    }

    // 重载赋值运算符
    MySharedPtr<T>& operator=(const MySharedPtr<T>& other) {
        if (this != &other) {
            (*refCount)--;
            if (*refCount == 0) {
                delete data;
                delete refCount;
            }

            data = other.data;
            refCount = other.refCount;
            (*refCount)++;
        }
        return *this;
    }

    T* get() const {
        return data;
    }

    int use_count() const {
        return *refCount;
    }

private:
    T* data;       // 指向被管理对象的指针
    int* refCount;  // 引用计数
};

int main() {
    MySharedPtr<int> ptr1(new int(42));
    MySharedPtr<int> ptr2 = ptr1; // 复制构造函数，引用计数+1

    std::cout << "ptr1 use count: " << ptr1.use_count() << std::endl;
    std::cout << "ptr2 use count: " << ptr2.use_count() << std::endl;

    {
        MySharedPtr<int> ptr3 = ptr1; // 复制构造函数，引用计数+1
        std::cout << "ptr1 use count: " << ptr1.use_count() << std::endl;
        std::cout << "ptr2 use count: " << ptr2.use_count() << std::endl;
        std::cout << "ptr3 use count: " << ptr3.use_count() << std::endl;
    } // ptr3 超出作用域，引用计数-1

    std::cout << "ptr1 use count: " << ptr1.use_count() << std::endl;
    std::cout << "ptr2 use count: " << ptr2.use_count() << std::endl;

    return 0;
}
