#export,import
--sqlplus����
@C:\oraclexe\app\oracle\product\11.2.0\server\rdbms\admin\catexp.sql
--prompt����(cmd)
exp system/manager tables=(hr.emp,hr.dept) grants=y indexes=y --���� ��ο� export 
-- grants: db��ü�� ���� ������ export�� ������ ����
-- indexes: ���̺�鿡 ���� �ε����� export �� ������ ����
-- tables: export�� ���̺� 

exp hr/hr file=hr.demp tables=emp,dept rows=y compress=y 
-- file: ���ϸ� �� ���� ����,
-- tables: export�� ���̺� 
-- rows: ����� export �ɰ����� ����
-- compress: ���׸�Ʈ���� ����Ȯ�念����� ������������

exp hr/hr file=hr_all.dmp owner=hr grants=y rows=y compress=y --hr������ ��� �����Ͱ� export
-- owner: export �� ����� ���

exp hr/hr full=y file=dba.dmp grants=y rows=y --��ü �����ͺ��̽��� export
-- full: ��ü �����ͺ��̽��� export ���� ����

imp system/manager file=exdat.dmp fromuser=hr touser=system tables=(dept,emp) -- import
-- file: import�� ���ϸ�
-- fromuser: export dump���Ϸκ��� �������� �ϴ� ��ü���� �����ִ� ����� ���
--> system���� Ȯ��

imp system/manager file=hr_all.dmp fromuser=hr touser=tiger --import\
--> tiger���� Ȯ��

#1. vline/vline88 ������ ����
create user vline
identified by vline88
default tablespace test temporary tablespace temp;
grant connect, resource to vline;
#2. ������ vline�� ������ ���� �Ӽ����� ���ǵ� ��� ���̺�(s_custom)�� ����
create table s_custom(
id number(7) not null,
 name varchar2(50) not null,
 phone varchar2(25),
 address varchar2(50),
 zip_code varchar2(10),
 region_id number(7),
 comments varchar2(100),
 credit_rating varchar2(9));
#3. �� ���̺� ������ 3���� �Է�
insert into s_custom(id,name,phone,address,zip_code,region_id,comments,credit_rating)
values (112234,'����','010-1111-2222','hi123@gmail.com','123412311',910102,null,'1���');
insert into s_custom(id,name,phone,address,zip_code,region_id,comments,credit_rating) 
values (143114,'��ο�','010-6617-7222','hello993@naver.com','1099211',673712,'����','3���');
insert into s_custom(id,name,phone,address,zip_code,region_id,comments,credit_rating) 
values (182934,'�����','010-7817-3982','gauza11@daum.net','996123',88122,'�����','2���');
#4. ������ ����
update s_custom
 set credit_rating='1���'
 where id=143114;
#5. ������ ����
delete from s_custom
 where id=143114;
#6. vline�� s_custom ���̺��� �����͸� �����޾Ƽ� tiger�� import
exp vline/vline88 tables=s_custom grants=y indexes=y
imp system/manager file=expdat.dmp fromuser=vline touser=tiger tables=(s_custom);

#loader �ǽ�
--���̺� �����
create table dept1 as
 select * from dept
 where 1=0;
 --���̺� ������ �δ�
 --cmdâ����
 sqlldr userid=hr/hr control='C:\Loader�ǽ�\demo1.ctl' log=demo1.log --demo1.ctl���ִ� ������ �δ�
 select * from dept1; --�����Ͱ� �δ��� ���� Ȯ��
 
 --emp���̺� �����ͷδ�
 sqlldr userid=hr/hr control='C:\Loader�ǽ�\demo2.ctl' data='C:\Loader�ǽ�\demo2.dat' log=demo2.log
-- control: ��Ʈ�� �����̸�
-- data: �Է� ������ ����
-- log: �α� ���� �̸�
select * from emp;


--���̺� ����
delete from emp;
create table emp_rating(
  empno number(4),
  leadership number(3),
  membershop number(3),
  english number(3),
  computing number(3));
--�δ�
sqlldr userid=hr/hr control=demo5.ctl log=demo2.log
--Ȯ��
select * from emp;
select * from emp_rating;

