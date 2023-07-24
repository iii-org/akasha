FROM dockerhub/library/php:7.4.33-cli-alpine3.16

RUN apk update && apk upgrade
COPY app /var/www/html
WORKDIR /var/www/html
EXPOSE 80
CMD ["php", "-S", "0.0.0.0:80"]
