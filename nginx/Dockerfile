# Set nginx base image
FROM nginx

MAINTAINER David Hevron

COPY nginx.conf /etc/nginx/nginx.conf

CMD ["nginx", "-g", "daemon off;"]
#CMD [ "service", "nginx", "start" ]