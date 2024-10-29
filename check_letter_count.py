def count_words_in_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            words = content.split()
            word_count = len(words)
            return word_count
    except FileNotFoundError:
        return f"Error: The file '{filename}' was not found."
    except UnicodeDecodeError:
        return f"Error: Could not decode the file '{filename}'. Please check the file encoding."

if __name__ == "__main__":
    filename = input("Enter the filename (including .txt extension): ")
    word_count = count_words_in_file(filename)
    print(f"The word count in '{filename}' is: {word_count}")
