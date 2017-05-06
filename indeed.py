import os
import multiprocessing
import string
from multiprocessing import Process
from multiprocessing import Queue
from bs4 import BeautifulSoup
from urllib.request import urlopen

class spider:

    def __init__(self,job,action,threads,n_job_postings=0):
        self.job = job
        self.threads = threads
        self.queue = Queue()
        self.n_job_postings = n_job_postings

        print("starting crawler")
        if action == "links":
            self.create_project_dir()
            worker = Process(target=self.get_links)
            worker.start()

        elif action == "texts":
            self.get_links_from_file()
            workers = []

            for i in range(threads):
                p = Process(target=self.worker,name=self.job.lower()+str(i))
                p.start()
                workers.append(p)







    def worker(self):
        counter = 0
        while self.queue.empty()==False:
            url = self.queue.get()
            self.get_text(multiprocessing.current_process().name,url,counter)
            counter+=1



    #get the text from each job posting and writes it onto the new file
    def get_text(self,thread,url,counter):
        # get html file
        print("Thread " + thread + " crawling " + url)
        html_string = self.get_hmtl_string(url)


        if html_string != None:
            soup = BeautifulSoup(html_string, "html.parser")
            text = " "
            for body in soup.find("body"):
                for p in soup.find_all("p"):
                    if "function()" not in p.get_text():
                        text += p.get_text() + "\n"

                for li in soup.find_all("li"):
                    if "function()" not in li.get_text() and "%" not in li.get_text():
                        text += li.get_text() + "\n"


            name = multiprocessing.current_process().name+str(counter)
            self.append_text_to_file(str(text),name)










    def get_links(self):
        start_url = "https://www.indeed.com/jobs?q=" + self.getJobTitle() + "&l="
        counter = 0
        print(start_url)


        while counter < self.n_job_postings:
            #get html file
            html_string = self.get_hmtl_string(start_url)
            soup = BeautifulSoup(html_string, "html.parser")



            # search for all links and save them
            for tag in soup.find_all("td"):
                if tag.get("id") == "resultsCol":
                    for h in tag.find_all("h2"):
                        for class_tag in h.get("class"):
                            if class_tag == "jobtitle":
                                for a in h.find_all("a"):
                                    for rel in a.get("rel"):
                                         if rel == "nofollow":
                                            link = "https://www.indeed.com" + a.get("href")
                                            self.append_to_file(link)
                                            counter+=1
            print(counter)

            #get the next page
            for tag in soup.find_all("div"):
                if tag.has_attr("class"):
                    for class_tag in tag.get("class"):
                        if class_tag == "pagination":
                            links = tag.find_all("a")
                            if links[4].get("href") != None:
                                start_url = "https://www.indeed.com" + links[4].get("href")
                                print(counter)
                            else:
                                loop = False
                                print("Got " + counter + " links. Thats it!")





    #get the html file from the url
    def get_hmtl_string(self,url):
        try:
            response = urlopen(url)
            html_bytes = response.read()
            html_string = html_bytes.decode("utf-8")

            return html_string
        except:
            return None


    #creates the project dir for the job
    def create_project_dir(self):
        if not os.path.exists(self.job):
            print("Creating project " + self.job)
            os.makedirs(self.job)

    #creates new file for each job posting
    def append_text_to_file(self,text,name):
        with open(self.job+"/" + name + ".txt", "w") as file:
            try:
                file.write(text)
            except:
                print(text)


            file.close()

    # add link onto data file
    def append_to_file(self,link):
        file = self.job +"/links.txt"
        with open(file, "a")as file:
            file.write(link + "\n")

    #reads the links file for the specified job and returns it
    def get_links_from_file(self):
        with open(self.job+"/links.txt","r") as file:
            links = file.read().split("\n")
            for link in links:
                self.queue.put(link)

    #aggregates the job titel for indeed.com search and returns it
    def getJobTitle(self):
        words = self.job.lower().split(" ")
        preparedTitle = ""
        for i in range(0,len(words)):
            if i != len(words)-1:
                preparedTitle += words[i] + "+"
            else:
                preparedTitle += words[i]

        return preparedTitle



