create table address
(id number(30),
name varchar2(50),
addr varchar2(100),
phone varchar2(30),
email varchar2(100));

insert into address
values(1,'HGDONG','SEOUL','123-4567','gdhong@cwunet.ac.kr');

create table addr_second(id,name,addr,phone,e_mail)
as select * from address;
create tabvle addr_fourth
as select id,name from address
where 1=2;

create table addr_third
as select id,name from address;

desc address;
desc addr_third;

select * from addr_third;

alter table address
add(birth date);

alter table address
add(comments varchar2(200) default 'No Comment');
desc address;

alter table address  drop column comments;
desc address;

alter table address
modify phone varchar2(50);

alter table address
modify phone varchar2(5);

desc address;

rename addr_second to client_address;

select * from tab;
drop table addr_third;
select * from tab
where tname='addr_third';

select * from client_address;
truncate table client_address;
select * from client_address;
rollback;
select * from client_address;
commit;
delete from student;
select * from student;
rollback;
select * from student;

comment on table address
is '�� �ּҷ��� �����ϱ� ���� ���̺�';

comment on column address.name
is '�� �̸�';

select comments
from user_tab_comments
where table_name='ADDRESS';

select * from user_col_comments
where table_name='ADDRESS';

comment on table ADDRESS IS '';
comment on column address.name is'';

select table_name from user_tables;
select owner, table_name from all_tables;

select owner,table_name from dba_tables;

select table_name,tablespace_name,min_extents,max_extents
from user_tables
where table_name like 'ADDR%';

select object_name,object_type,created
from user_objects
where object_name like 'ADDR%' and object_type='TABLE';

select * from user_catalog;

create table subject
(subno NUMBER(5)
    constraint subject_no_pk primary key
    deferrable initially deferred
    using index tablespace indx,
subname varchar2(20)
    constraint subject_name_nn not null,
term varchar2(1)
    constraint subject_term_ck check (term in('1','2')),
type varchar2(1));

conn /as sysdba
grant create tablespace to hr;
conn hr/hr;
create tablespace indx
datafile 'C:\oraclexe\app\oracle\oradata\XE\indx.dbf' size 100m;

alter table student
add constraint stud_no_pk primary key(studno);
select constraint_name,constraint_type
from user_constraints;
select constraint_name,constraint_type
from user_constraints

where table_name in('SUBJECT','STUDENT');
desc student;

alter table student
add constraint stud_no_pk primary key(studno);

create table sugang
(studno number(5)
    constraint sugang_studno_fk references student(studno),
subno number(5)
    constraint sugang_subno_fk references subject(subno),
regdate Date,
result number(3),
    constraint sugang_pk primary key(studno,subno));
desc sugang;

alter table student
add constraint stud_idnum_uk unique(idnum);
alter table student
modify(name constraint stud_name_nn Not null);
alter table student add constraint stud_deptno_fk
foreign key(deptno) references department(deptno);
desc student;
commit;

alter table department
add constraint deptno_no_pk primary key(deptno);
alter table department modify(dname not null);
ALTER TABLE PROFESSOR ADD CONSTRAINTS PROF_PK PRIMARY KEY(PROFNO);
ALTER TABLE professor modify(name not null);
select deptno from professor minus select deptno from department;
select * from professor;
delete from professor where deptno=103;
alter table professor add constraints prof_fk
foreign key(deptno) references department(deptno);

insert into subject values(1,'SQL','1','�ʼ�');
insert into subject values(3,'java','3','����');
desc subject;
select * from subject;
insert into subject values(4,'�����ͺ��̽�','1','1');
insert into subject values(4,'�����͸𵨸�','2','1');
commit;

select constraint_name,constraint_type,deferrable,deferred from
user_constraints
where table_name='SUBJECT';

select constraint_name,constraint_type
from user_constraints
where table_name='SUBJECT';

alter table subject
drop constraint subject_term_ck;

select constraint_name,constraint_type
from user_constraints
where table_name='SUBJECT';

alter table sugang
disable constraint sugang_pk;
alter table sugang
disable constraint sugang_studno_fk;
select constraint_name,status
from user_constraints
where table_name in('SUGANG','SUBJECT');

alter table sugang
enable constraint sugang_pk;
alter table sugang
enable constraint sugang_studno_fk;
select constraint_name,status
from user_constraints
where table_name='SUGANG';

select table_name,constraint_name,constraint_type,statUs
from user_constraints
where table_name IN('STUDENT','PROFESSOR','DEPARTMENT');

#1. 20�� �μ� ��� �������� ���� �����ȣ,�����,�μ���ȣ�� �����ϴ� EMP20���̺� ����
select * from emp;
create table emp20 as
select empno,ename,deptno from emp
where deptno=20;
#2. 1���� ������ emp20 ���̺��� ename �ʵ带 10-->30���� ����
desc emp20;
alter table emp20
modify (ename varchar2(30));
#3. emp20 ���̺� number(8,2)�� salary �÷��� �߰�
alter table emp20
add (SALARY number(8,2));
#4. emp20 ���̺��� �����͸� �����ϼ���.
delete from emp20;
#5. emp20 �����͸� �ٽ� rollback ��Ű����.
rollback;
select * from emp20;
#6. emp20 ���̺��� �����Ϳ� �Ҵ�� ������ ����
truncate table emp20;
#7. department ���̺��� �⺻ Ű ���� ���Ǳ� ���� �������� ���� �������� 300���� ����,�ִ밪�� 20000,10�� �����ϴ� ������, �̸��� department_id_seq
create sequence department_id_seq
increment by 10
start with 300
maxvalue 20000;
select min_value,max_value,increment_by,last_number
from user_sequences
where sequence_name='DEPARTMENT_ID_SEQ';
#8. 7���� ������ �������� nextval�� �̿��ؼ� department ���̺� mis,management �μ��� �߰�
select * from department;
insert into department values
(department_id_seq.nextval,'MIS',null,null);
insert into department values
(department_id_seq.nextval,'MANAGEMENT',null,null);
#9. DEPARTMENT ���̺��� 310�� �μ����� MARKETING���� ����
update department set dname='MARKETING'
where deptno=310;
select * from department;

#1. customer ���̺��� �Ʒ��� ���� �����ϼ���.(��, CUSTOMER_ID PRIMARY KEY,name NOT NULL ����, deptno�� dept���̺��� deptno�� �����ϵ���)
CREATE table customer(
CUSTOMER_ID number(7) PRIMARY KEY,
NAME varchar2(25) NOT NULL,
PHONE varchar2(25),
ADDRESS varchar2(20),
deptno number(4) 
constraint FK_CUSTOMER references dept(deptno));
desc customer;

create unique index idxd_dept_name
on department(dname);

create index idx_stud_birthdate
on student(birthdate);

create index idx_stud_dno_grade
on student(deptno,grade);

create index fidx_stud_no_name on student(deptno desc,name ASC);

create index uppercase_idx ON emp (UPPER(ename));

create index idx_standard_weight ON student((height-100)*0.9);


--set Autotrace ���--SQL PLUS

