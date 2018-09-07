#1. 학생 테이블에서 이름이 '진'으로끝나고,지도교수가 배정되지않는 101번 학과 학생의 아이디,이름,학년,학과 번호를 출력
desc student;
desc professor;
select * from student;
select * from professor;
select userid,name,grade,deptno
from student
where name like '%진'
and profno is null
and deptno=101;
#3. 학생 테이블과 교수테이블에 대한 union연산을 수행하여 name,userid,sal를 출력, sal없으면 0출력 
select name,userid,0
from student
union
select name,userid,sal
from professor;
#4. 교수 테이블에서 전체 교수의 급여를 인상하기 위한 직원 명단을 출력(단 직급이 전임강사인 사람들은 명단에서 제외)
select name,position
from professor
where not position='전임강사';

select name,position
from professor
minus
select name,position
from professor
where position='전임강사';

#4. 교수 테이블에서 교수번호,이름,직위,급여 그리고 19%증가된 급여를 이름,직위순으로 출력(증가된급여의 열 레이블은 newsalary)
select profno,name,position,sal,sal*1.19 New_Salary
from professor
order by name,position;

#사원테이블에서 'scott'사원의 이름,부서번호,급여를 출력
select ename,deptno,sal
from emp
where INITCAP(ename)='scott';

select empno,ename,job
from emp
where lower(job)='manager';

#이름이 A,T,M으로 시작하는 모든 사원에 대해서 첫번째문자는 대문자 나머지는 소문자로 나타나는 사원의 이름과 이름길이를 출력
select initcap(ename),Length(ename) "Length"
from emp
where ename like 'A%'
or ename like 'T%'
or ename like 'M%';

#1. 여러분이 출생한지 며칠째인지 출력
select to_date(19950924,'yyyy/mm/dd')
from dual;
select trunc(sysdate - to_date(19950924,'yyyy/mm/dd')) "lived day"
from dual;


#1. 업무(job)가 manager이거나 salesman이며 급여가 $1500,3000,5000이 아닌 모든 사원에 대해서 이름 업무 그리고 급여를출력
select * from emp;
select ename,job,sal
from emp
where job in ('MANAGER','SALESMAN')
and sal not in(1500,3000,5000);
#2. 사원 테이블에서 사원번호,이름,직업,급여 그리고 22%증가된 급여를 이름,직업순으로 출력
select empno,ename,job,sal,sal*1.22 "New Salary"
from emp
order by ename ,job;
#3. 2에 추가하여 새로운 급여에서 예전의 급여(sal)를 뺴는 열을 추가
select empno,ename,job,sal,sal*1.22 "New Salary",sal*1.22-sal Increase
from emp
order by ename ,job;
#4. 사원의 이름과 보너스를 출력하는 질의를 작성. 보너스안받으면 no commission출력
select ename,NVL(to_char(comm),'No Commission') comm
from emp;
