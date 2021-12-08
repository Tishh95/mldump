FROM python
EXPOSE 5000
COPY . .
RUN pip install -r requirements.txt
CMD python test.py
