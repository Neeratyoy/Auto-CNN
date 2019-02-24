
if __name__=="__main__":
    print("1. Run simple BOHB")
    print("2. Run Transfer Learning")
    print("3. Run Transfer Config")
    print("4. Run default")
    print("5. Exit")
    try:
        option = int(input("Select choice (1-5): "))
    except:
        option = 0

    while option!=5:
        if option>0 and option<5:
            print("\nYou have selected: ", option)
            if option==1:
                # Do BOHB
                print("Hello")
            elif option==2:
                # Transfer Learning
                print("World")
            elif option==3:
                # Transfer Config
                print("!")
            else:
                # Run default
                print("Hmm")
        print("~+~"*40)
        print("\nKindly enter only integers between 1 and 4 -> {1, 2, 3, 4}\n")
        print("1. Run simple BOHB")
        print("2. Run Transfer Learning")
        print("3. Run Transfer Config")
        print("4. Run default")
        print("5. Exit")
        try:
            option = int(input("Select choice (1-5): "))
        except:
            option = 0


    print("\nByee! ^_^")
    exit()
