-- id;name;emailHash;reputation;location;creationDate

CREATE TABLE IF NOT EXISTS Hashes (
    Id           int PRIMARY KEY,
    name         varchar(260),
    emailHash    varchar(32),
    reputation   int,
    creationDate timestamp not NULL
);

