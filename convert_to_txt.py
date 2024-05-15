from bs4 import BeautifulSoup

def html_to_txt(html_file, output_file):
    
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()
    
    print(html_content[:1000])
    # Find the start and end index of the first <p> tag
    start_index = html_content.find('<p>')
    end_index = html_content.find('</p>', start_index) + len('</p>')
    html_content = html_content[:start_index] + html_content[end_index:]
   
    soup = BeautifulSoup(html_content, 'html.parser')
    

    for title in soup.find_all('title'):
        title.extract()
    # print(soup)
    
    first_p = soup.find('p')
        
    for p in soup.find_all('p'):
        p.string = p.get_text().replace('\n', ' ')
    
    text = soup.get_text()
    
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    # print(text)
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)


html_to_txt('scripts/0101.html', 'output.txt')

