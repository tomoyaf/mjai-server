FROM ruby:2.4.1-alpine3.6

LABEL maintainer="tomoyaf"

ENV MJAI_HOME /mjai
ENV MJAI_SOURCE https://github.com/mahjong-server/mahjong-server/archive/master.tar.gz

EXPOSE 11600

RUN apk update && \
    apk upgrade && \
    apk add --no-cache build-base libxml2-dev libxslt-dev openssl ca-certificates wget libxml2 libxslt && \
    update-ca-certificates && \
    gem install nokogiri -- --use-system-libraries --with-xml2-include=/usr/include/libxml2 --with-xml2- lib=/usr/lib/ && \
    gem install websocket-client-simple bundler sass && \
    mkdir -p $MJAI_HOME && \
    wget -O - $MJAI_SOURCE | tar zxf - mahjong-server-master/mjai -C $MJAI_HOME --strip-components=2

WORKDIR $MJAI_HOME

CMD ["/bin/sh", "-c", "ruby ${MJAI_HOME}/bin/multisrv.rb"]
