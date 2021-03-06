FROM ruby:2.4.1-alpine3.6

LABEL maintainer="tomoyaf"

ENV MJAI_HOME /mjai
ENV MJAI_SOURCE https://github.com/tomoyaf/mahjong-server/archive/master.tar.gz

EXPOSE 11600

RUN apk update && \
    apk upgrade && \
    apk add --no-cache build-base libxml2-dev libxslt-dev openssl ca-certificates wget && \
    update-ca-certificates && \
    mkdir -p $MJAI_HOME && \
    wget -O - $MJAI_SOURCE | tar zxf - mahjong-server-master/mjai -C $MJAI_HOME --strip-components=2

WORKDIR $MJAI_HOME

RUN bundle install

CMD ["/bin/sh", "-c", "ruby ${MJAI_HOME}/bin/multisrv.rb"]
