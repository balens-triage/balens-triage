-- instead of managing multiple tables, we are using 'ns' to namespace within a table

CREATE TABLE IF NOT EXISTS commit_logs (
    ns         varchar(150) not null,
    commit     varchar(40)  not null,
    message    text,
    dev_name   varchar(400),
    dev_email  varchar(300),
    bug_ids    text [],
    created_at timestamp,
    primary key (commit, ns)
);

CREATE TABLE IF NOT EXISTS developers (
    ns                 varchar(150) not null,
    dev_user_name      varchar(100) not null primary key,
    issue_count        int,
    is_invalid         SMALLINT,
    is_inactive        SMALLINT,
    web_url            varchar(1000),
    name               varchar(105),
    real_name          varchar(255),
    email              varchar(105),
    groups             varchar(105),
    id                 INT,
    avg_fix_time       float,
    newcomer_fix_time  float,
    newcomer_last_date timestamp,
    profi_first_date   timestamp
);

CREATE TABLE IF NOT EXISTS issues (
    ns             varchar(150) not null,
    id             int          not null,
    summary        text,
    title          text         not null,
    assignee       varchar(150),
    assignee_email varchar(150),
    cc_list        text,
    keywords       text,
    created_at     timestamp,
    modified_at    timestamp,
    priority       text,
    severity       text,
    classification varchar(45),
    product        varchar(245),
    component      varchar(245),
    platform       varchar(65),
    status         varchar(45),
    primary key (id, ns)
);


CREATE TABLE IF NOT EXISTS tossing (
    ns     varchar(150) not null,
    bug_id int          not null,
    email  varchar(100),
    time   timestamp,
    primary key (ns, bug_id, email, time)
);

CREATE TABLE IF NOT EXISTS resolution (
    ns     varchar(150) not null,
    bug_id int          not null,
    status varchar(100),
    time   timestamp,
    primary key (ns, bug_id, status, time)
);


CREATE INDEX IF NOT EXISTS issues_email
    on issues (assignee_email);

CREATE TABLE IF NOT EXISTS comments (
    ns         varchar(150) not null,
    id         int          not null,
    bug_id     int          not null,
    count      int,
    creator    varchar(100) not null,
    created_at timestamp,
    text       text,
    primary key (id, ns)
);
